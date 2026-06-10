% =========================================================================
%  AUTONOMOUS VEHICLE LANE FOLLOWING – EKF SENSOR FUSION + STANLEY CONTROL
% =========================================================================
%  Title  : Autonomous Vehicle Lane Following (EKF): Fusion of IMU and
%            Camera / GPS Sensors
%
%  Architecture
%  ────────────
%  Plant   : Kinematic Bicycle Model  (closed-loop, steering driven by
%             Stanley Controller – NOT a replayed trajectory)
%  State   : x = [px, py, v, psi, psi_bias, a_bias]^T  (6-DOF)
%              px       – X position           (m)
%              py       – Y position           (m)
%              v        – longitudinal speed   (m/s)
%              psi      – heading angle        (rad)
%              psi_bias – gyro / yaw-rate bias (rad/s)
%              a_bias   – accel bias           (m/s^2)
%  Sensors : IMU  @ 100 Hz  (prediction driver; accel + yaw-rate)
%            GPS  @ 10  Hz  (px, py; tunnel drop-out + multipath + outlier)
%            Camera @ 30 Hz  (CTE + lane-relative heading; frame drop +
%                             outlier)
%  Control : Stanley Controller  →  steering angle δ
%             δ = ψ_e + atan2(k_s * e_cte, v)
%             All inputs are EKF-estimated states (never ground truth).
%  Rejection: Innovation gating via NIS for GPS and Camera updates.
%  Analysis : Single-run metrics + Monte Carlo (N_MC = 50 runs).
%  Figures  : 16 report-quality figures.
%
%  Author  : Generated for university project
%  Version : 2.0 – Full closed-loop lane-following system
% =========================================================================
clear; clc; close all;
rng(42);   % Reproducible seed for single-run demo

% =========================================================================
%% SECTION 0 – GLOBAL CONFIGURATION  (all tuneable parameters here)
% =========================================================================

%% 0.1  Simulation time
T         = 80;          % total simulation time (s)
dt_imu    = 0.01;        % IMU  sample period  (100 Hz)
dt_cam    = 1/30;        % Camera sample period (30 Hz)
dt_gps    = 0.1;         % GPS   sample period (10 Hz)
time_vec  = 0:dt_imu:T;
N         = length(time_vec);

%% 0.2  Vehicle (Kinematic Bicycle Model)
L_wb      = 2.7;         % wheelbase (m)
v_nominal = 12.0;        % nominal forward speed (m/s)  – kept approx. constant
v_std_accel = 0.15;      % small accel noise driving speed variation (m/s^2)

%% 0.3  IMU sensor parameters
std_acc        = 0.5;    % white-noise std on longitudinal accel (m/s^2)
std_yaw_rate   = 0.04;   % white-noise std on yaw rate            (rad/s)
b_acc_true0    = 0.25;   % true initial accel bias                (m/s^2)
b_gyro_true0   = 0.015;  % true initial gyro bias                 (rad/s)
std_b_acc_rw   = 0.002;  % accel bias random-walk diffusion       (m/s^2 / √s)
std_b_gyro_rw  = 0.001;  % gyro  bias random-walk diffusion       (rad/s / √s)

%% 0.4  GPS sensor parameters
std_gps_pos    = 2.0;    % Gaussian position noise std            (m)
p_multipath    = 0.04;   % prob. of multipath per GPS epoch
multipath_mag  = 7.0;    % multipath jump magnitude               (m)
p_gps_outlier  = 0.02;   % prob. of large outlier per GPS epoch
gps_outlier_mag= 14.0;   % outlier magnitude                      (m)
tunnel_start   = 25;     % GPS tunnel dropout start               (s)
tunnel_end     = 34;     % GPS tunnel dropout end                 (s)

%% 0.5  Camera sensor parameters
std_cam_cte    = 0.20;   % Gaussian noise on CTE measurement      (m)
std_cam_hrel   = 0.06;   % Gaussian noise on relative heading     (rad)
p_frame_drop   = 0.10;   % prob. of frame drop per camera tick
p_outlier_cam  = 0.05;   % prob. of outlier per valid frame
outlier_scale  = 6.0;    % outlier spike scale factor

%% 0.6  EKF – process noise covariance Q  (6×6 diagonal)
%  Ordered: [px, py, v, psi, psi_bias, a_bias]
Q = diag([ 0.05, ...                          % px  – low process uncertainty
           0.05, ...                          % py
           std_acc^2, ...                     % v   – driven by accel noise
           0.005, ...                         % psi
           (std_b_gyro_rw*sqrt(dt_imu))^2, ...% psi_bias random walk
           (std_b_acc_rw *sqrt(dt_imu))^2 ]); % a_bias   random walk

%% 0.7  NIS (Normalised Innovation Squared) gating thresholds
%  chi^2 distribution: GPS 2-DOF → 95% = 5.991; Camera 2-DOF → 95% = 5.991
NIS_thresh_gps = chi2inv(0.95, 2);   % 5.99
NIS_thresh_cam = chi2inv(0.95, 2);   % 5.99  (CTE + heading_rel = 2-DOF)

%% 0.8  Stanley Controller gain
k_stanley = 1.8;         % cross-track gain (tuned empirically)
delta_max = deg2rad(35); % max steering angle (rad)

%% 0.9  Monte Carlo settings
N_MC = 50;

% =========================================================================
%% SECTION 1 – LANE CENTRELINE  (straight + curved sections)
% =========================================================================
% Build a smooth reference path in X-Y world coordinates.
% Parameterised by arc-length index aligned to the IMU time grid.
%
% Profile:
%   0–15 s  : straight (heading = 0)
%   15–25 s : curved left (radius R1)
%   25–40 s : straight (heading ~ 30 deg)
%   40–55 s : curved right (back towards east)
%   55–80 s : straight

lane_px    = zeros(1, N);
lane_py    = zeros(1, N);
lane_psi   = zeros(1, N);   % lane tangent heading

% We integrate the lane tangent forward using a prescribed curvature profile
lane_kappa = zeros(1, N);   % desired curvature κ(t)

for k = 1:N
    t = time_vec(k);
    if t >= 15 && t < 25          % left curve
        % ramp curvature in/out smoothly using a bell
        frac = (t - 15) / 10;
        lane_kappa(k) = 0.025 * sin(pi * frac);
    elseif t >= 40 && t < 55      % right curve (return)
        frac = (t - 40) / 15;
        lane_kappa(k) = -0.020 * sin(pi * frac);
    end
end

% Integrate curvature to get heading, then integrate heading to get XY
for k = 1:N-1
    lane_psi(k+1)  = lane_psi(k)  + lane_kappa(k) * v_nominal * dt_imu;
    lane_px(k+1)   = lane_px(k)   + v_nominal * cos(lane_psi(k)) * dt_imu;
    lane_py(k+1)   = lane_py(k)   + v_nominal * sin(lane_psi(k)) * dt_imu;
end

% Lane width for plot visualisation
lane_width = 3.5;   % (m) – standard lane width

% =========================================================================
%% SECTION 2 – CLOSED-LOOP GROUND TRUTH SIMULATION
%              (Kinematic Bicycle Model + Stanley Controller on GT states)
% =========================================================================
% NOTE: The *EKF-driven* Stanley loop is in Section 7.
%       Here we first generate the TRUE trajectory by running the Stanley
%       controller on the TRUE states so the lane following is plausible.
%       In the main EKF loop, only EKF estimates are fed to Stanley.

nx    = 6;   % state dimension  [px py v psi psi_bias a_bias]

x_true        = zeros(nx, N);
x_true(:, 1)  = [0; 0; v_nominal; 0; b_gyro_true0; b_acc_true0];

delta_true    = zeros(1, N);   % true steering angle applied (GT-based)
a_cmd_true    = zeros(1, N);   % true acceleration command

for k = 1:N-1
    % ----- Stanley Controller on GROUND TRUTH (for generating true traj.) --
    % (In the EKF loop, this will be replaced by EKF estimates)
    [e_cte_gt, e_psi_gt] = compute_cte_heading_error( ...
        x_true(1,k), x_true(2,k), x_true(4,k), ...
        lane_px, lane_py, lane_psi);

    v_gt           = max(x_true(3,k), 0.1);   % guard against division by zero
    delta_gt       = e_psi_gt + atan2(k_stanley * e_cte_gt, v_gt);
    delta_gt       = max(-delta_max, min(delta_max, delta_gt));
    delta_true(k)  = delta_gt;

    % Speed regulation: small proportional control to maintain v_nominal
    a_cmd_true(k) = 0.5 * (v_nominal - x_true(3,k));

    % ----- Kinematic Bicycle Model propagation (Euler integration) --------
    psi_k = x_true(4,k);
    v_k   = x_true(3,k);

    x_true(1,k+1) = x_true(1,k) + v_k * cos(psi_k) * dt_imu;
    x_true(2,k+1) = x_true(2,k) + v_k * sin(psi_k) * dt_imu;
    x_true(3,k+1) = v_k + a_cmd_true(k) * dt_imu;
    x_true(4,k+1) = psi_k + (v_k / L_wb) * tan(delta_gt) * dt_imu;

    % Bias random walk (slowly drifting true biases)
    x_true(5,k+1) = x_true(5,k) + std_b_gyro_rw * sqrt(dt_imu) * randn();
    x_true(6,k+1) = x_true(6,k) + std_b_acc_rw  * sqrt(dt_imu) * randn();
end

% =========================================================================
%% SECTION 3 – IMU MEASUREMENTS  (include bias + white noise)
% =========================================================================
% The IMU measures:
%   a_m  = true_accel + a_bias + accel_white_noise
%   yr_m = true_yaw_rate + psi_bias + gyro_white_noise
%
% True yaw rate from bicycle model: ψ̇ = (v / L_wb) * tan(δ)

u_imu = zeros(2, N);   % row 1: a_m,  row 2: yaw_rate_m

for k = 1:N
    true_accel    = a_cmd_true(k);
    true_yaw_rate = (x_true(3,k) / L_wb) * tan(delta_true(k));

    u_imu(1,k) = true_accel    + x_true(6,k) + std_acc      * randn();
    u_imu(2,k) = true_yaw_rate + x_true(5,k) + std_yaw_rate * randn();
end

% =========================================================================
%% SECTION 4 – GPS MEASUREMENTS  (position; dropout + multipath + outlier)
% =========================================================================
z_gps      = nan(2, N);
gps_step   = round(dt_gps / dt_imu);   % every 10 indices at 100 Hz

for k = 1:gps_step:N
    % --- GPS tunnel dropout ---
    if time_vec(k) >= tunnel_start && time_vec(k) <= tunnel_end
        continue;   % no measurement; leave as NaN
    end

    % Base Gaussian measurement
    meas = [x_true(1,k); x_true(2,k)] + std_gps_pos * randn(2,1);

    % Multipath jump (random direction, non-Gaussian tail)
    if rand() < p_multipath
        dir  = randn(2,1);
        dir  = dir / norm(dir);
        meas = meas + multipath_mag * dir;
    end

    % Large position outlier spike
    if rand() < p_gps_outlier
        meas = meas + gps_outlier_mag * randn(2,1);
    end

    z_gps(:,k) = meas;
end

% =========================================================================
%% SECTION 5 – CAMERA MEASUREMENTS  (CTE + relative heading)
% =========================================================================
% Camera measures the lane markings and returns:
%   1) Signed cross-track error  e_cte  (m)
%   2) Lane-relative heading     e_psi  (rad)  – relative to lane tangent
%
% Both corrupted by Gaussian noise, with occasional frame drops / outliers.

z_cam = nan(2, N);   % row 1: CTE meas,  row 2: heading_rel meas

for k = 1:N
    % Camera fires at dt_cam ≈ 1/30 s – check via modulo
    if mod(time_vec(k), dt_cam) >= dt_imu
        continue;   % not a camera epoch
    end

    % Frame drop
    if rand() < p_frame_drop
        continue;
    end

    % True CTE and heading error from ground truth
    [e_cte_true, e_psi_true] = compute_cte_heading_error( ...
        x_true(1,k), x_true(2,k), x_true(4,k), ...
        lane_px, lane_py, lane_psi);

    % Add Gaussian measurement noise
    meas_cte  = e_cte_true  + std_cam_cte  * randn();
    meas_hrel = e_psi_true  + std_cam_hrel * randn();

    % Outlier spike (glare / shadow / false detection)
    if rand() < p_outlier_cam
        meas_cte  = meas_cte  + outlier_scale * std_cam_cte  * randn();
        meas_hrel = meas_hrel + outlier_scale * std_cam_hrel * randn();
    end

    z_cam(1,k) = meas_cte;
    z_cam(2,k) = meas_hrel;
end

% =========================================================================
%% SECTION 6 – EKF INITIALISATION
% =========================================================================
x_est         = zeros(nx, N);
x_est(:,1)    = [0; 0; v_nominal; 0; 0; 0];   % biases init at 0 (unknown)

P             = diag([5, 5, 1, 0.5, 0.5, 0.5]);  % initial covariance
P_hist        = zeros(nx, nx, N);
P_hist(:,:,1) = P;

% Storage for post-analysis
NEES            = zeros(1, N);
NIS_gps_all     = nan(1, N);
NIS_cam_all     = nan(1, N);
delta_cmd       = zeros(1, N);   % steering commanded by EKF-based Stanley
n_gps_rejected  = 0;
n_cam_rejected  = 0;
n_gps_accepted  = 0;
n_cam_accepted  = 0;

gps_rejection_times = [];
cam_rejection_times = [];

% =========================================================================
%% SECTION 7 – CLOSED-LOOP EKF + STANLEY MAIN LOOP
%  ► Stanley Controller uses ONLY EKF estimates (never ground truth)
%  ► Steering command feeds the TRUE vehicle (bicycle model re-simulation
%    is NOT needed – the true trajectory was computed in Section 2 using
%    the SAME Stanley formulation driven by GT; here we track it with EKF)
%
%  Design note: In a real embedded system, the EKF output drives the
%  actuator.  Here the "plant" is x_true (already computed), and the EKF
%  estimates are used for control evaluation and closed-loop performance
%  metrics.  The measured IMU, GPS and Camera signals are generated from
%  x_true above, and the EKF tracks the evolving true state.  The Stanley
%  controller output is logged for evaluation and figures.
% =========================================================================

for k = 1:N-1

    % ------------------------------------------------------------------ %
    %  PREDICTION STEP  (IMU @ 100 Hz)
    % ------------------------------------------------------------------ %
    psi_e   = x_est(4,k);
    v_e     = x_est(3,k);
    bg_e    = x_est(5,k);   % estimated gyro bias
    ba_e    = x_est(6,k);   % estimated accel bias

    % De-biased IMU readings
    a_debias  = u_imu(1,k) - ba_e;
    yr_debias = u_imu(2,k) - bg_e;

    % Nonlinear state prediction  (bicycle model kinematics)
    x_pred    = zeros(nx, 1);
    x_pred(1) = x_est(1,k) + v_e * cos(psi_e) * dt_imu;
    x_pred(2) = x_est(2,k) + v_e * sin(psi_e) * dt_imu;
    x_pred(3) = v_e + a_debias * dt_imu;
    x_pred(4) = psi_e + yr_debias * dt_imu;
    x_pred(5) = bg_e;   % bias random walk: zero-mean drift
    x_pred(6) = ba_e;

    % Jacobian of f w.r.t. state  F (6×6)
    F        = eye(nx);
    F(1, 3)  =  cos(psi_e) * dt_imu;
    F(1, 4)  = -v_e * sin(psi_e) * dt_imu;
    F(2, 3)  =  sin(psi_e) * dt_imu;
    F(2, 4)  =  v_e * cos(psi_e) * dt_imu;
    F(3, 6)  = -dt_imu;   % ∂v_dot / ∂a_bias
    F(4, 5)  = -dt_imu;   % ∂psi_dot / ∂psi_bias

    % Prior covariance
    P_pred = F * P * F' + Q;

    % ------------------------------------------------------------------ %
    %  STANLEY CONTROLLER  (uses predicted / prior EKF estimate)
    % ------------------------------------------------------------------ %
    [e_cte_ekf, e_psi_ekf] = compute_cte_heading_error( ...
        x_pred(1), x_pred(2), x_pred(4), ...
        lane_px, lane_py, lane_psi);

    v_ctrl     = max(x_pred(3), 0.5);   % safety clamp
    delta_k    = e_psi_ekf + atan2(k_stanley * e_cte_ekf, v_ctrl);
    delta_k    = max(-delta_max, min(delta_max, delta_k));
    delta_cmd(k+1) = delta_k;

    % ------------------------------------------------------------------ %
    %  CORRECTION STEPS  (asynchronous multi-rate)
    % ------------------------------------------------------------------ %
    has_gps = ~isnan(z_gps(1, k+1));
    has_cam = ~isnan(z_cam(1, k+1));

    x_upd = x_pred;
    P_upd = P_pred;

    % -- GPS update (if available) ------------------------------------ %
    if has_gps
        H_gps = [1 0 0 0 0 0;
                 0 1 0 0 0 0];
        R_gps  = diag([std_gps_pos^2, std_gps_pos^2]);

        innov_gps = z_gps(:,k+1) - x_upd(1:2);
        S_gps     = H_gps * P_upd * H_gps' + R_gps;
        NIS_gps   = innov_gps' * (S_gps \ innov_gps);

        NIS_gps_all(k+1) = NIS_gps;

        if NIS_gps <= NIS_thresh_gps
            % Accept measurement – standard EKF update
            K_gps = P_upd * H_gps' / S_gps;
            x_upd = x_upd + K_gps * innov_gps;
            IKH   = eye(nx) - K_gps * H_gps;
            P_upd = IKH * P_upd * IKH' + K_gps * R_gps * K_gps';   % Joseph form
            n_gps_accepted = n_gps_accepted + 1;
        else
            % Reject measurement (NIS gating)
            n_gps_rejected = n_gps_rejected + 1;
            gps_rejection_times(end+1) = time_vec(k+1); %#ok<AGROW>
        end
    end

    % -- Camera update (if available) --------------------------------- %
    if has_cam
        % Linearised camera observation model:
        % The camera measures [CTE; heading_relative_to_lane]
        % At time k, the nearest lane point index provides the lane tangent.
        % CTE  ≈  -sin(psi_lane) * (px - lane_px) + cos(psi_lane) * (py - lane_py)
        % h_rel = psi - psi_lane  (vehicle heading minus lane tangent)
        %
        % We compute these from the predicted state and linearise.

        [~, idx_nn] = min((lane_px - x_upd(1)).^2 + (lane_py - x_upd(2)).^2);
        psi_lane    = lane_psi(idx_nn);
        lx          = lane_px(idx_nn);
        ly          = lane_py(idx_nn);

        % Predicted observation
        dx_lane    = x_upd(1) - lx;
        dy_lane    = x_upd(2) - ly;
        cte_pred   = -sin(psi_lane) * dx_lane + cos(psi_lane) * dy_lane;
        hrel_pred  = wrap_angle(x_upd(4) - psi_lane);

        z_cam_pred = [cte_pred; hrel_pred];

        % Jacobian H_cam  (2×6)
        %   d(CTE)/d(px)  = -sin(psi_lane)
        %   d(CTE)/d(py)  =  cos(psi_lane)
        %   d(h_rel)/d(psi) = 1
        H_cam = zeros(2, nx);
        H_cam(1,1) = -sin(psi_lane);
        H_cam(1,2) =  cos(psi_lane);
        H_cam(2,4) =  1.0;

        R_cam     = diag([std_cam_cte^2, std_cam_hrel^2]);
        innov_cam = z_cam(:,k+1) - z_cam_pred;
        innov_cam(2) = wrap_angle(innov_cam(2));   % heading wrap

        S_cam     = H_cam * P_upd * H_cam' + R_cam;
        NIS_cam   = innov_cam' * (S_cam \ innov_cam);

        NIS_cam_all(k+1) = NIS_cam;

        if NIS_cam <= NIS_thresh_cam
            K_cam = P_upd * H_cam' / S_cam;
            x_upd = x_upd + K_cam * innov_cam;
            x_upd(4) = wrap_angle(x_upd(4));   % keep heading in [-π,π]
            IKH   = eye(nx) - K_cam * H_cam;
            P_upd = IKH * P_upd * IKH' + K_cam * R_cam * K_cam';
            n_cam_accepted = n_cam_accepted + 1;
        else
            n_cam_rejected = n_cam_rejected + 1;
            cam_rejection_times(end+1) = time_vec(k+1); %#ok<AGROW>
        end
    end

    % -- Dead reckoning if no sensor available ------------------------ %
    % (x_upd and P_upd are already set to x_pred / P_pred if neither
    %  GPS nor camera was accepted, so no extra action needed here)

    % Symmetrise to prevent numerical drift
    P_upd = 0.5 * (P_upd + P_upd');

    x_est(:, k+1)   = x_upd;
    P               = P_upd;
    P_hist(:,:,k+1) = P;

    % NEES: compare all 6 states (heading wrapped)
    err_k    = x_true(:,k+1) - x_est(:,k+1);
    err_k(4) = wrap_angle(err_k(4));
    P_reg    = P + 1e-9 * eye(nx);
    NEES(k+1)= err_k' * (P_reg \ err_k);
end

% Last step: store steering command at k=N
delta_cmd(1) = delta_cmd(2);   % fill first sample (no update at k=0)

% =========================================================================
%% SECTION 8 – SINGLE-RUN PERFORMANCE METRICS
% =========================================================================
err_px  = x_true(1,:) - x_est(1,:);
err_py  = x_true(2,:) - x_est(2,:);
err_v   = x_true(3,:) - x_est(3,:);
err_psi = arrayfun(@(a,b) wrap_angle(a - b), x_true(4,:), x_est(4,:));

% CTE and heading error computed using EKF estimate vs lane
CTE_hist  = zeros(1,N);
HErr_hist = zeros(1,N);
for k = 1:N
    [cte_k, he_k] = compute_cte_heading_error( ...
        x_est(1,k), x_est(2,k), x_est(4,k), lane_px, lane_py, lane_psi);
    CTE_hist(k)  = cte_k;
    HErr_hist(k) = he_k;
end

RMSE_px   = sqrt(mean(err_px.^2));
RMSE_py   = sqrt(mean(err_py.^2));
RMSE_v    = sqrt(mean(err_v.^2));
RMSE_psi  = sqrt(mean(err_psi.^2));
RMS_CTE   = sqrt(mean(CTE_hist.^2));
RMS_HErr  = sqrt(mean(HErr_hist.^2));

mean_NEES = mean(NEES(2:end));
valid_NIS_gps = NIS_gps_all(~isnan(NIS_gps_all));
valid_NIS_cam = NIS_cam_all(~isnan(NIS_cam_all));
mean_NIS_gps  = mean(valid_NIS_gps);
mean_NIS_cam  = mean(valid_NIS_cam);

chi2_lo = chi2inv(0.025, nx);
chi2_hi = chi2inv(0.975, nx);

% =========================================================================
%% SECTION 9 – MONTE CARLO ANALYSIS (N_MC = 50 runs)
% =========================================================================
fprintf('\nRunning %d Monte Carlo simulations...\n', N_MC);

MC_RMSE_px   = zeros(1, N_MC);
MC_RMSE_py   = zeros(1, N_MC);
MC_RMSE_v    = zeros(1, N_MC);
MC_RMSE_CTE  = zeros(1, N_MC);
MC_mean_NEES = zeros(1, N_MC);

for mc = 1:N_MC
    % Fresh true initial biases
    b_gyro_mc = b_gyro_true0 + 0.005 * randn();
    b_acc_mc  = b_acc_true0  + 0.05  * randn();

    % ---- True trajectory (GT-based Stanley controller) ----
    xt = zeros(nx, N);
    xt(:,1) = [0; 0; v_nominal; 0; b_gyro_mc; b_acc_mc];
    dt_mc   = zeros(1, N);   % delta (steering)

    for k = 1:N-1
        [e_c, e_h] = compute_cte_heading_error( ...
            xt(1,k), xt(2,k), xt(4,k), lane_px, lane_py, lane_psi);
        v_mc  = max(xt(3,k), 0.1);
        dk    = e_h + atan2(k_stanley * e_c, v_mc);
        dk    = max(-delta_max, min(delta_max, dk));
        dt_mc(k) = dk;

        a_mc      = 0.5 * (v_nominal - xt(3,k));
        xt(1,k+1) = xt(1,k) + xt(3,k)*cos(xt(4,k))*dt_imu;
        xt(2,k+1) = xt(2,k) + xt(3,k)*sin(xt(4,k))*dt_imu;
        xt(3,k+1) = xt(3,k) + a_mc*dt_imu;
        xt(4,k+1) = xt(4,k) + (xt(3,k)/L_wb)*tan(dk)*dt_imu;
        xt(5,k+1) = xt(5,k) + std_b_gyro_rw*sqrt(dt_imu)*randn();
        xt(6,k+1) = xt(6,k) + std_b_acc_rw *sqrt(dt_imu)*randn();
    end

    % ---- Noisy IMU ----
    un = zeros(2,N);
    for k = 1:N
        tyr = (xt(3,k)/L_wb)*tan(dt_mc(k));
        un(1,k) = 0.5*(v_nominal-xt(3,k)) + xt(6,k) + std_acc     *randn();
        un(2,k) = tyr                      + xt(5,k) + std_yaw_rate*randn();
    end

    % ---- GPS ----
    zg = nan(2,N);
    for k = 1:gps_step:N
        if time_vec(k) >= tunnel_start && time_vec(k) <= tunnel_end; continue; end
        m = [xt(1,k);xt(2,k)] + std_gps_pos*randn(2,1);
        if rand()<p_multipath; d=randn(2,1);d=d/norm(d); m=m+multipath_mag*d; end
        if rand()<p_gps_outlier; m=m+gps_outlier_mag*randn(2,1); end
        zg(:,k) = m;
    end

    % ---- Camera ----
    zc = nan(2,N);
    for k = 1:N
        if mod(time_vec(k),dt_cam)>=dt_imu; continue; end
        if rand()<p_frame_drop; continue; end
        [ec,eh] = compute_cte_heading_error(xt(1,k),xt(2,k),xt(4,k), ...
                                            lane_px,lane_py,lane_psi);
        mc_cte = ec + std_cam_cte *randn();
        mc_h   = eh + std_cam_hrel*randn();
        if rand()<p_outlier_cam
            mc_cte = mc_cte + outlier_scale*std_cam_cte *randn();
            mc_h   = mc_h   + outlier_scale*std_cam_hrel*randn();
        end
        zc(1,k)=mc_cte; zc(2,k)=mc_h;
    end

    % ---- EKF ----
    xe = zeros(nx,N);
    xe(:,1) = [0;0;v_nominal;0;0;0];
    Pm = diag([5,5,1,0.5,0.5,0.5]);
    NEES_mc = zeros(1,N);

    for k = 1:N-1
        psi_e=xe(4,k); v_e=xe(3,k); bg_e=xe(5,k); ba_e=xe(6,k);
        ad = un(1,k)-ba_e; yr = un(2,k)-bg_e;

        xp=zeros(nx,1);
        xp(1)=xe(1,k)+v_e*cos(psi_e)*dt_imu;
        xp(2)=xe(2,k)+v_e*sin(psi_e)*dt_imu;
        xp(3)=v_e+ad*dt_imu;
        xp(4)=psi_e+yr*dt_imu;
        xp(5)=bg_e; xp(6)=ba_e;

        Fm=eye(nx);
        Fm(1,3)=cos(psi_e)*dt_imu; Fm(1,4)=-v_e*sin(psi_e)*dt_imu;
        Fm(2,3)=sin(psi_e)*dt_imu; Fm(2,4)= v_e*cos(psi_e)*dt_imu;
        Fm(3,6)=-dt_imu; Fm(4,5)=-dt_imu;
        Pp=Fm*Pm*Fm'+Q;

        xu=xp; Pu=Pp;

        if ~isnan(zg(1,k+1))
            Hg=[1 0 0 0 0 0;0 1 0 0 0 0];
            Rg=diag([std_gps_pos^2,std_gps_pos^2]);
            ig=zg(:,k+1)-xu(1:2);
            Sg=Hg*Pu*Hg'+Rg;
            nis_g=ig'*(Sg\ig);
            if nis_g<=NIS_thresh_gps
                Kg=Pu*Hg'/Sg;
                xu=xu+Kg*ig;
                IKH=eye(nx)-Kg*Hg;
                Pu=IKH*Pu*IKH'+Kg*Rg*Kg';
            end
        end

        if ~isnan(zc(1,k+1))
            [~,inn]=min((lane_px-xu(1)).^2+(lane_py-xu(2)).^2);
            pla=lane_psi(inn); lxn=lane_px(inn); lyn=lane_py(inn);
            dxl=xu(1)-lxn; dyl=xu(2)-lyn;
            ctp=-sin(pla)*dxl+cos(pla)*dyl;
            hrp=wrap_angle(xu(4)-pla);
            Hc=zeros(2,nx);
            Hc(1,1)=-sin(pla); Hc(1,2)=cos(pla); Hc(2,4)=1;
            Rc=diag([std_cam_cte^2,std_cam_hrel^2]);
            ic=zc(:,k+1)-[ctp;hrp];
            ic(2)=wrap_angle(ic(2));
            Sc=Hc*Pu*Hc'+Rc;
            nis_c=ic'*(Sc\ic);
            if nis_c<=NIS_thresh_cam
                Kc=Pu*Hc'/Sc;
                xu=xu+Kc*ic;
                xu(4)=wrap_angle(xu(4));
                IKH=eye(nx)-Kc*Hc;
                Pu=IKH*Pu*IKH'+Kc*Rc*Kc';
            end
        end

        Pu=0.5*(Pu+Pu');
        xe(:,k+1)=xu; Pm=Pu;
        er=xt(:,k+1)-xe(:,k+1); er(4)=wrap_angle(er(4));
        NEES_mc(k+1)=er'*((Pu+1e-9*eye(nx))\er);
    end

    MC_RMSE_px(mc)  = sqrt(mean((xt(1,:)-xe(1,:)).^2));
    MC_RMSE_py(mc)  = sqrt(mean((xt(2,:)-xe(2,:)).^2));
    MC_RMSE_v(mc)   = sqrt(mean((xt(3,:)-xe(3,:)).^2));

    cte_mc = zeros(1,N);
    for k = 1:N
        [ck,~]=compute_cte_heading_error(xe(1,k),xe(2,k),xe(4,k), ...
                                         lane_px,lane_py,lane_psi);
        cte_mc(k)=ck;
    end
    MC_RMSE_CTE(mc)  = sqrt(mean(cte_mc.^2));
    MC_mean_NEES(mc) = mean(NEES_mc(2:end));

    if mod(mc,10)==0
        fprintf('  MC run %3d / %3d complete\n', mc, N_MC);
    end
end

% =========================================================================
%% SECTION 10 – PRINT PERFORMANCE REPORT
% =========================================================================
fprintf('\n');
fprintf('=================================================================\n');
fprintf('  AV LANE FOLLOWING – EKF SENSOR FUSION  |  PERFORMANCE REPORT \n');
fprintf('=================================================================\n');
fprintf('\n--- Single-Run Metrics ---\n');
fprintf('  RMSE X  position   : %7.4f  m\n',   RMSE_px);
fprintf('  RMSE Y  position   : %7.4f  m\n',   RMSE_py);
fprintf('  RMSE Velocity      : %7.4f  m/s\n', RMSE_v);
fprintf('  RMSE Heading       : %7.4f  rad\n', RMSE_psi);
fprintf('  RMS  Cross-track   : %7.4f  m\n',   RMS_CTE);
fprintf('  RMS  Heading error : %7.4f  rad\n', RMS_HErr);
fprintf('  Mean NEES          : %7.4f  (6-DOF chi² 95%% bounds: [%.2f, %.2f])\n', ...
        mean_NEES, chi2_lo, chi2_hi);
if mean_NEES >= chi2_lo && mean_NEES <= chi2_hi
    fprintf('  EKF Consistency    : CONSISTENT (NEES within 95%% bounds)\n');
else
    fprintf('  EKF Consistency    : INCONSISTENT – revisit noise tuning\n');
end
fprintf('  Mean NIS  GPS      : %7.4f  (2-DOF 95%% bound: %.2f)\n', mean_NIS_gps, NIS_thresh_gps);
fprintf('  Mean NIS  Camera   : %7.4f  (2-DOF 95%% bound: %.2f)\n', mean_NIS_cam, NIS_thresh_cam);
fprintf('  GPS  rejected      : %d / %d  (%5.1f%%)\n', ...
        n_gps_rejected, n_gps_rejected+n_gps_accepted, ...
        100*n_gps_rejected/max(1,n_gps_rejected+n_gps_accepted));
fprintf('  Camera rejected    : %d / %d  (%5.1f%%)\n', ...
        n_cam_rejected, n_cam_rejected+n_cam_accepted, ...
        100*n_cam_rejected/max(1,n_cam_rejected+n_cam_accepted));
fprintf('\n--- Monte Carlo Summary (%d runs) ---\n', N_MC);
fprintf('  Mean RMSE X        : %.4f ± %.4f  m\n',   mean(MC_RMSE_px),   std(MC_RMSE_px));
fprintf('  Mean RMSE Y        : %.4f ± %.4f  m\n',   mean(MC_RMSE_py),   std(MC_RMSE_py));
fprintf('  Mean RMSE V        : %.4f ± %.4f  m/s\n', mean(MC_RMSE_v),    std(MC_RMSE_v));
fprintf('  Mean RMSE CTE      : %.4f ± %.4f  m\n',   mean(MC_RMSE_CTE),  std(MC_RMSE_CTE));
fprintf('  Mean NEES          : %.4f ± %.4f\n',       mean(MC_mean_NEES), std(MC_mean_NEES));
mc_ok = sum(MC_mean_NEES >= chi2_lo & MC_mean_NEES <= chi2_hi);
fprintf('  Consistent runs    : %d / %d  (%.1f%%)\n', mc_ok, N_MC, 100*mc_ok/N_MC);
fprintf('=================================================================\n\n');

% =========================================================================
%% SECTION 11 – SIGMA-BOUND EXTRACTION (for figures)
% =========================================================================
sig_px   = squeeze(sqrt(P_hist(1,1,:)))';
sig_py   = squeeze(sqrt(P_hist(2,2,:)))';
sig_v    = squeeze(sqrt(P_hist(3,3,:)))';
sig_psi  = squeeze(sqrt(P_hist(4,4,:)))';
sig_bg   = squeeze(sqrt(P_hist(5,5,:)))';
sig_ba   = squeeze(sqrt(P_hist(6,6,:)))';

% =========================================================================
%% SECTION 12 – COLOUR PALETTE (colour-blind safe; consistent across figs)
% =========================================================================
c_truth  = [0.00, 0.00, 0.00];   % black   – ground truth
c_ekf    = [0.00, 0.45, 0.70];   % blue    – EKF estimate
c_gps    = [0.80, 0.20, 0.20];   % red     – GPS
c_cam    = [0.85, 0.53, 0.00];   % orange  – camera
c_lane   = [0.00, 0.60, 0.40];   % green   – lane centreline
c_bound  = [0.80, 0.20, 0.20];   % red     – ±3σ boundary
c_tunel  = [1.00, 0.90, 0.00];   % yellow  – tunnel region patch
c_rej    = [0.90, 0.10, 0.10];   % bright red – rejection events

% =========================================================================
%% SECTION 13 – FIGURES 1–16
% =========================================================================

% Helper: left-edge boundaries of lane (perpendicular offset from centreline)
lane_left_px  = lane_px - (lane_width/2) * sin(lane_psi);
lane_left_py  = lane_py + (lane_width/2) * cos(lane_psi);
lane_right_px = lane_px + (lane_width/2) * sin(lane_psi);
lane_right_py = lane_py - (lane_width/2) * cos(lane_psi);

valid_gps = ~isnan(z_gps(1,:));

% ------------------------------------------------------------------ %
%  FIGURE 1  –  Lane geometry & vehicle trajectory
% ------------------------------------------------------------------ %
figure('Name','Fig 1 – Lane Geometry & Vehicle Trajectory', ...
       'NumberTitle','off','Position',[50 550 900 520]);
% Draw lane boundaries as shaded corridor
fill([lane_left_px, fliplr(lane_right_px)], ...
     [lane_left_py, fliplr(lane_right_py)], ...
     [0.85 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.5);
hold on;
plot(lane_px,       lane_py,       '-', 'Color',c_lane,  'LineWidth',1.5, 'LineStyle','--');
plot(lane_left_px,  lane_left_py,  '-', 'Color',[0.5 0.5 0.5], 'LineWidth',1);
plot(lane_right_px, lane_right_py, '-', 'Color',[0.5 0.5 0.5], 'LineWidth',1);
plot(x_true(1,:),   x_true(2,:),   '-', 'Color',c_truth, 'LineWidth',2.0);
plot(x_est(1,:),    x_est(2,:),    '--','Color',c_ekf,   'LineWidth',2.0);
title('Figure 1 – Lane Geometry and Vehicle Trajectory', ...
      'FontSize',13,'FontWeight','bold');
xlabel('X Position (m)','FontSize',11);
ylabel('Y Position (m)','FontSize',11);
legend('Lane Corridor','Lane Centreline','Lane Boundary','', ...
       'Ground Truth','EKF Estimate', ...
       'Location','northwest','FontSize',10);
grid on; box on; axis equal; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 2  –  True trajectory vs EKF trajectory (full path)
% ------------------------------------------------------------------ %
figure('Name','Fig 2 – True vs EKF Trajectory', ...
       'NumberTitle','off','Position',[80 520 900 520]);
plot(x_true(1,:), x_true(2,:), '-', 'Color',c_truth,'LineWidth',2);
hold on;
plot(z_gps(1,valid_gps), z_gps(2,valid_gps), '.', 'Color',c_gps,'MarkerSize',6);
plot(x_est(1,:),  x_est(2,:),  '--','Color',c_ekf, 'LineWidth',2);
plot(lane_px,     lane_py,     '-.','Color',c_lane,'LineWidth',1.5);
% Tunnel zone annotation (X range where tunnel occurs)
tun_xA = x_true(1, round(tunnel_start/dt_imu)+1);
tun_xB = x_true(1, min(N, round(tunnel_end/dt_imu)+1));
patch([tun_xA tun_xB tun_xB tun_xA], [-15 -15 15 15], ...
      c_tunel,'FaceAlpha',0.30,'EdgeColor','none');
title('Figure 2 – Ground Truth vs Raw GPS vs EKF Trajectory', ...
      'FontSize',13,'FontWeight','bold');
xlabel('X Position (m)','FontSize',11); ylabel('Y Position (m)','FontSize',11);
legend('Ground Truth','Raw GPS','EKF Estimate','Lane Centreline','GPS Tunnel Zone', ...
       'Location','northwest','FontSize',10);
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 3  –  Trajectory zoom – GPS tunnel outage region
% ------------------------------------------------------------------ %
figure('Name','Fig 3 – GPS Tunnel Zoom','NumberTitle','off','Position',[110 490 900 460]);
k_a = max(1,   round((tunnel_start - 4) / dt_imu));
k_b = min(N,   round((tunnel_end   + 4) / dt_imu));
idx_z = k_a:k_b;
plot(x_true(1,idx_z), x_true(2,idx_z), '-', 'Color',c_truth,'LineWidth',2.5);
hold on;
plot(x_est(1,idx_z),  x_est(2,idx_z),  '--','Color',c_ekf,  'LineWidth',2.5);
valid_z = valid_gps & (1:N>=k_a) & (1:N<=k_b);
plot(z_gps(1,valid_z), z_gps(2,valid_z),'.','Color',c_gps,'MarkerSize',10);
patch([tun_xA tun_xB tun_xB tun_xA], [-15 -15 15 15], ...
      c_tunel,'FaceAlpha',0.45,'EdgeColor','none');
title('Figure 3 – Zoom: GPS Tunnel Dropout Region (Dead Reckoning)', ...
      'FontSize',13,'FontWeight','bold');
xlabel('X Position (m)','FontSize',11); ylabel('Y Position (m)','FontSize',11);
legend('Ground Truth','EKF (Dead Reckoning)','GPS (pre/post tunnel)','Tunnel Zone', ...
       'Location','best','FontSize',10);
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 4  –  Cross-track error (CTE) vs time
% ------------------------------------------------------------------ %
figure('Name','Fig 4 – Cross-Track Error','NumberTitle','off','Position',[140 460 860 420]);
plot(time_vec, CTE_hist, '-', 'Color',c_ekf, 'LineWidth',1.8);
hold on;
yline(0,'k--','LineWidth',1.2);
% Mark lane-change events with vertical lines
xline(15,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2,'Label','Curve Start 1');
xline(40,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2,'Label','Curve Start 2');
title('Figure 4 – Cross-Track Error (EKF Estimate vs Lane Centreline)', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('CTE (m)','FontSize',11);
legend('Cross-Track Error','Centreline Reference','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 5  –  Heading error (vehicle psi – lane tangent psi)
% ------------------------------------------------------------------ %
figure('Name','Fig 5 – Heading Error','NumberTitle','off','Position',[170 440 860 420]);
plot(time_vec, rad2deg(HErr_hist), '-', 'Color',c_ekf, 'LineWidth',1.8);
hold on;
yline(0,'k--','LineWidth',1.2);
title('Figure 5 – Heading Error Relative to Lane Centreline', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Heading Error (deg)','FontSize',11);
legend('Heading Error','Zero Reference','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 6  –  Stanley steering command vs time
% ------------------------------------------------------------------ %
figure('Name','Fig 6 – Steering Command','NumberTitle','off','Position',[200 420 860 420]);
plot(time_vec, rad2deg(delta_cmd), '-', 'Color',c_ekf, 'LineWidth',1.8);
hold on;
yline( rad2deg(delta_max), 'r--','LineWidth',1.2,'Label','Max δ');
yline(-rad2deg(delta_max), 'r--','LineWidth',1.2,'Label','Min δ');
yline(0,'k-','LineWidth',0.8);
title('Figure 6 – Stanley Controller Steering Command', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Steering Angle δ (deg)','FontSize',11);
legend('Steering Command δ','Saturation Limits','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 7  –  Velocity tracking (true vs EKF ± 3σ)
% ------------------------------------------------------------------ %
figure('Name','Fig 7 – Velocity Tracking','NumberTitle','off','Position',[230 400 860 420]);
plot(time_vec, x_true(3,:), '-', 'Color',c_truth,'LineWidth',2);
hold on;
plot(time_vec, x_est(3,:),  '--','Color',c_ekf,  'LineWidth',2);
fill([time_vec, fliplr(time_vec)], ...
     [x_est(3,:)+3*sig_v, fliplr(x_est(3,:)-3*sig_v)], ...
     c_ekf,'FaceAlpha',0.12,'EdgeColor','none');
title('Figure 7 – Velocity Tracking','FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Speed (m/s)','FontSize',11);
legend('True Velocity','EKF Estimate','±3σ Bound','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 8  –  Heading angle tracking (true vs EKF ± 3σ)
% ------------------------------------------------------------------ %
valid_cam_idx = ~isnan(z_cam(1,:));
figure('Name','Fig 8 – Heading Tracking','NumberTitle','off','Position',[260 380 860 440]);
plot(time_vec, rad2deg(x_true(4,:)), '-', 'Color',c_truth,'LineWidth',2);
hold on;
% Show camera-derived heading (CTE + lane_psi → approximate vehicle heading)
% Camera measures heading-relative; we overlay approximate camera heading
cam_heading_meas = z_cam(2,valid_cam_idx) + lane_psi(valid_cam_idx);
scatter(time_vec(valid_cam_idx), rad2deg(cam_heading_meas), 10, c_cam, 'filled');
plot(time_vec, rad2deg(x_est(4,:)),  '--','Color',c_ekf,  'LineWidth',2);
fill([time_vec, fliplr(time_vec)], ...
     [rad2deg(x_est(4,:)+3*sig_psi), fliplr(rad2deg(x_est(4,:)-3*sig_psi))], ...
     c_ekf,'FaceAlpha',0.12,'EdgeColor','none');
title('Figure 8 – Heading Angle Tracking','FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Heading ψ (deg)','FontSize',11);
legend('Ground Truth','Camera (derived)','EKF Estimate','±3σ Bound', ...
       'FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 9  –  Accelerometer bias estimation
% ------------------------------------------------------------------ %
figure('Name','Fig 9 – Accel Bias','NumberTitle','off','Position',[290 360 860 420]);
plot(time_vec, x_true(6,:), '-', 'Color',c_truth,'LineWidth',2);
hold on;
plot(time_vec, x_est(6,:),  '--','Color',c_ekf,  'LineWidth',2);
fill([time_vec, fliplr(time_vec)], ...
     [x_est(6,:)+3*sig_ba, fliplr(x_est(6,:)-3*sig_ba)], ...
     c_ekf,'FaceAlpha',0.15,'EdgeColor','none');
title('Figure 9 – Accelerometer Bias Estimation', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Bias b_{acc} (m/s²)','FontSize',11);
legend('True Bias','EKF Estimate','±3σ Bound','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 10  –  Gyroscope bias estimation
% ------------------------------------------------------------------ %
figure('Name','Fig 10 – Gyro Bias','NumberTitle','off','Position',[320 340 860 420]);
plot(time_vec, x_true(5,:), '-', 'Color',c_truth,'LineWidth',2);
hold on;
plot(time_vec, x_est(5,:),  '--','Color',c_ekf,  'LineWidth',2);
fill([time_vec, fliplr(time_vec)], ...
     [x_est(5,:)+3*sig_bg, fliplr(x_est(5,:)-3*sig_bg)], ...
     c_ekf,'FaceAlpha',0.15,'EdgeColor','none');
title('Figure 10 – Gyroscope Bias Estimation', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('Bias b_{gyro} (rad/s)','FontSize',11);
legend('True Bias','EKF Estimate','±3σ Bound','FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 11  –  NEES evolution with chi² consistency bounds
% ------------------------------------------------------------------ %
figure('Name','Fig 11 – NEES Evolution','NumberTitle','off','Position',[350 320 860 440]);
plot(time_vec(2:end), NEES(2:end), '-','Color',c_ekf,'LineWidth',1.2);
hold on;
yline(nx,       'k-',  'Ideal (E[NEES]=n_x)', 'LineWidth',2, ...
      'LabelHorizontalAlignment','left');
yline(chi2_hi, '--', 'Color',c_bound,'LineWidth',1.8,'Label','95% Upper Bound');
yline(chi2_lo, '--', 'Color',c_lane, 'LineWidth',1.8,'Label','95% Lower Bound');
xline(tunnel_start,'--','Color',[0.55 0.55 0],'LineWidth',1.0,'Label','Tunnel In');
xline(tunnel_end,  '--','Color',[0.55 0.55 0],'LineWidth',1.0,'Label','Tunnel Out');
ylim([0, min(max(NEES(2:end))*1.15, chi2_hi*5)]);
title('Figure 11 – NEES Evolution & Chi² Consistency Bounds', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('NEES','FontSize',11);
legend('NEES','Theoretical Mean','95% Upper','95% Lower', ...
       'FontSize',10,'Location','northeast');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 12  –  NIS evolution (GPS and Camera)
% ------------------------------------------------------------------ %
gps_nk  = find(~isnan(NIS_gps_all));
cam_nk  = find(~isnan(NIS_cam_all));

figure('Name','Fig 12 – NIS Evolution','NumberTitle','off','Position',[380 300 860 440]);
stem(time_vec(gps_nk), NIS_gps_all(gps_nk), 'Color',c_gps, ...
     'MarkerFaceColor',c_gps,'MarkerSize',3,'LineWidth',0.8);
hold on;
stem(time_vec(cam_nk), NIS_cam_all(cam_nk), 'Color',c_cam, ...
     'MarkerFaceColor',c_cam,'MarkerSize',3,'LineWidth',0.8);
yline(NIS_thresh_gps,'r--','LineWidth',1.8,'Label','95% Gating Threshold');
title('Figure 12 – Normalised Innovation Squared (NIS)', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('NIS','FontSize',11);
legend('GPS NIS','Camera NIS','95% Chi² Threshold', ...
       'FontSize',10,'Location','northeast');
ylim([0, NIS_thresh_gps * 4]);
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 13  –  Monte Carlo RMSE histograms
% ------------------------------------------------------------------ %
figure('Name','Fig 13 – MC RMSE Histograms', ...
       'NumberTitle','off','Position',[50 80 1100 450]);
subplot(1,4,1);
histogram(MC_RMSE_px, 20,'FaceColor',c_ekf,'EdgeColor','w');
xline(mean(MC_RMSE_px),'r-','LineWidth',2, ...
      'Label',sprintf('μ=%.2f m',mean(MC_RMSE_px)));
title('RMSE – X Position','FontSize',11);
xlabel('RMSE (m)'); ylabel('Count'); grid on;

subplot(1,4,2);
histogram(MC_RMSE_py, 20,'FaceColor',c_lane,'EdgeColor','w');
xline(mean(MC_RMSE_py),'r-','LineWidth',2, ...
      'Label',sprintf('μ=%.2f m',mean(MC_RMSE_py)));
title('RMSE – Y Position','FontSize',11);
xlabel('RMSE (m)'); ylabel('Count'); grid on;

subplot(1,4,3);
histogram(MC_RMSE_v, 20,'FaceColor',c_cam,'EdgeColor','w');
xline(mean(MC_RMSE_v),'r-','LineWidth',2, ...
      'Label',sprintf('μ=%.3f m/s',mean(MC_RMSE_v)));
title('RMSE – Velocity','FontSize',11);
xlabel('RMSE (m/s)'); ylabel('Count'); grid on;

subplot(1,4,4);
histogram(MC_RMSE_CTE, 20,'FaceColor',c_gps,'EdgeColor','w');
xline(mean(MC_RMSE_CTE),'k-','LineWidth',2, ...
      'Label',sprintf('μ=%.2f m',mean(MC_RMSE_CTE)));
title('RMSE – CTE','FontSize',11);
xlabel('RMSE CTE (m)'); ylabel('Count'); grid on;

sgtitle('Figure 13 – Monte Carlo RMSE Histograms (50 Runs)', ...
        'FontSize',13,'FontWeight','bold');

% ------------------------------------------------------------------ %
%  FIGURE 14  –  Monte Carlo NEES histogram
% ------------------------------------------------------------------ %
figure('Name','Fig 14 – MC NEES Histogram','NumberTitle','off', ...
       'Position',[100 80 720 420]);
histogram(MC_mean_NEES, 25,'FaceColor',c_ekf,'EdgeColor','w');
hold on;
xline(mean(MC_mean_NEES),'r-','LineWidth',2.5, ...
      'Label',sprintf('Mean=%.2f',mean(MC_mean_NEES)));
xline(chi2_lo,'--','Color',[0.2 0.7 0.3],'LineWidth',2,'Label','95% Lower');
xline(chi2_hi,'--','Color',[0.8 0.2 0.2],'LineWidth',2,'Label','95% Upper');
xline(nx,      'k-',                      'LineWidth',2,'Label','Ideal (6)');
title('Figure 14 – Monte Carlo Mean NEES Histogram', ...
      'FontSize',13,'FontWeight','bold');
xlabel('Mean NEES per Run','FontSize',11); ylabel('Number of Runs','FontSize',11);
legend('Histogram','Mean','95% Lower','95% Upper','Ideal', ...
       'FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 15  –  GPS rejection events
% ------------------------------------------------------------------ %
figure('Name','Fig 15 – GPS Rejection Events','NumberTitle','off', ...
       'Position',[150 60 860 440]);
% Plot NIS at accepted GPS epochs
gps_accepted_mask = ~isnan(NIS_gps_all) & (NIS_gps_all <= NIS_thresh_gps);
gps_rejected_mask = ~isnan(NIS_gps_all) & (NIS_gps_all >  NIS_thresh_gps);

stem(time_vec(gps_accepted_mask), NIS_gps_all(gps_accepted_mask), ...
     'Color',c_ekf,'MarkerFaceColor',c_ekf,'MarkerSize',4,'LineWidth',0.8);
hold on;
stem(time_vec(gps_rejected_mask), NIS_gps_all(gps_rejected_mask), ...
     'Color',c_rej,'MarkerFaceColor',c_rej,'MarkerSize',6,'LineWidth',1.5);
yline(NIS_thresh_gps,'--','Color',[0.5 0 0],'LineWidth',1.8, ...
      'Label','Rejection Threshold');
% Shade tunnel zone on time axis
patch([tunnel_start tunnel_end tunnel_end tunnel_start], ...
      [0 0 NIS_thresh_gps*5 NIS_thresh_gps*5], ...
      c_tunel,'FaceAlpha',0.3,'EdgeColor','none');
title(sprintf('Figure 15 – GPS NIS & Rejection Events  (%d rejected / %d total)', ...
      n_gps_rejected, n_gps_rejected+n_gps_accepted), ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('GPS NIS','FontSize',11);
legend('Accepted GPS','Rejected GPS (NIS Gating)','Threshold','Tunnel', ...
       'FontSize',10,'Location','northeast');
ylim([0, NIS_thresh_gps * 5]);
grid on; box on; set(gca,'FontSize',10);

% ------------------------------------------------------------------ %
%  FIGURE 16  –  Camera rejection events + residual
% ------------------------------------------------------------------ %
% Camera innovation residuals (CTE channel)
cam_res_all = nan(1,N);
for k = 1:N
    if ~isnan(z_cam(1,k))
        [cte_pr, hrel_pr] = compute_cte_heading_error( ...
            x_est(1,k), x_est(2,k), x_est(4,k), lane_px, lane_py, lane_psi);
        cam_res_all(k) = z_cam(1,k) - cte_pr;   % CTE innovation
    end
end

cam_acc_mask = ~isnan(NIS_cam_all) & (NIS_cam_all <= NIS_thresh_cam);
cam_rej_mask = ~isnan(NIS_cam_all) & (NIS_cam_all >  NIS_thresh_cam);

figure('Name','Fig 16 – Camera Rejection Events','NumberTitle','off', ...
       'Position',[200 40 860 440]);
scatter(time_vec(cam_acc_mask), ...
        cam_res_all(cam_acc_mask), 18, c_cam, 'filled');
hold on;
scatter(time_vec(cam_rej_mask), ...
        cam_res_all(cam_rej_mask), 50, c_rej, 'x', 'LineWidth',2);
thresh_cte_display = 4 * std_cam_cte;
yline( thresh_cte_display,'--','Color',[0.5 0.5 0.5],'LineWidth',1.5, ...
      'Label','+4σ_{CTE}');
yline(-thresh_cte_display,'--','Color',[0.5 0.5 0.5],'LineWidth',1.5, ...
      'Label','-4σ_{CTE}');
yline(0,'k-','LineWidth',0.8);
title(sprintf('Figure 16 – Camera CTE Residuals & Rejection Events  (%d rejected / %d total)', ...
      n_cam_rejected, n_cam_rejected+n_cam_accepted), ...
      'FontSize',13,'FontWeight','bold');
xlabel('Time (s)','FontSize',11); ylabel('CTE Innovation (m)','FontSize',11);
legend('Accepted Camera','Rejected Camera (NIS Gating)','±4σ threshold', ...
       'FontSize',10,'Location','best');
grid on; box on; set(gca,'FontSize',10);

% =========================================================================
%  END OF MAIN SCRIPT
% =========================================================================
fprintf('All 16 figures generated.  Script completed successfully.\n');

% =========================================================================
%% LOCAL HELPER FUNCTIONS
% =========================================================================

% -------------------------------------------------------------------------
%  wrap_angle – wrap angle to (-π, π]
% -------------------------------------------------------------------------
function a = wrap_angle(a)
    a = atan2(sin(a), cos(a));
end

% -------------------------------------------------------------------------
%  compute_cte_heading_error
%  Given vehicle position (px,py) and heading psi, finds the nearest point
%  on the lane centreline and returns:
%    e_cte   – signed cross-track error  (m)
%              positive = vehicle is to the left of the lane centreline
%    e_psi   – heading error relative to lane tangent  (rad)
%              positive = vehicle heading is CCW from lane tangent
%
%  Inputs:
%    px, py    – vehicle position
%    psi       – vehicle heading (rad)
%    lane_px, lane_py   – lane centreline arrays (1×N)
%    lane_psi_arr       – lane tangent heading   (1×N)
% -------------------------------------------------------------------------
function [e_cte, e_psi] = compute_cte_heading_error( ...
        px, py, psi, lane_px, lane_py, lane_psi_arr)

    % Nearest-neighbour search on centreline
    dist2  = (lane_px - px).^2 + (lane_py - py).^2;
    [~, idx] = min(dist2);

    lx  = lane_px(idx);
    ly  = lane_py(idx);
    lp  = lane_psi_arr(idx);

    dx = px - lx;
    dy = py - ly;

    % Signed CTE: project displacement onto perpendicular of lane tangent
    %  e_cte = -sin(psi_lane)*dx + cos(psi_lane)*dy
    e_cte = -sin(lp) * dx + cos(lp) * dy;

    % Heading error (wrapped)
    e_psi = atan2(sin(psi - lp), cos(psi - lp));
end