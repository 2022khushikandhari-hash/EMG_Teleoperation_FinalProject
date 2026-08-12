%% EMG Teleoperation — Robot Arm Simulation
%  Reads predictions.csv and animates a 4-DOF robot arm
%  Updated for new gesture labels: up, down, roll, yaw left, yaw right, bicep extend

clear; clc; close all;

%% ── BUILD THE ROBOT ─────────────────────────────────────────
robot = rigidBodyTree('DataFormat', 'column');

b1 = rigidBody('upper_arm');
j1 = rigidBodyJoint('shoulder', 'revolute');
setFixedTransform(j1, trvec2tform([0 0 0.10]));
j1.PositionLimits = [-pi/2, pi/2];
b1.Joint = j1;
addBody(robot, b1, 'base');

b2 = rigidBody('forearm');
j2 = rigidBodyJoint('elbow', 'revolute');
setFixedTransform(j2, trvec2tform([0.15 0 0]));
j2.PositionLimits = [-pi/2, pi/2];
b2.Joint = j2;
addBody(robot, b2, 'upper_arm');

b3 = rigidBody('wrist');
j3 = rigidBodyJoint('wrist_pitch', 'revolute');
setFixedTransform(j3, trvec2tform([0.10 0 0]));
j3.PositionLimits = [-pi/2, pi/2];
b3.Joint = j3;
addBody(robot, b3, 'forearm');

b4 = rigidBody('end_effector');
j4 = rigidBodyJoint('wrist_roll', 'revolute');
setFixedTransform(j4, trvec2tform([0.06 0 0]));
j4.PositionLimits = [-pi/2, pi/2];
b4.Joint = j4;
addBody(robot, b4, 'wrist');

fprintf('Robot built: %d joints\n', robot.NumBodies);

%% ── GESTURE TO JOINT ANGLE MAP ──────────────────────────────
% Angles in radians: [shoulder, elbow, wrist_pitch, wrist_roll]
% Updated for your new movement labels

gesture_map = containers.Map( ...
    {'up', 'down', 'bicep extend', 'roll', ...
     'yaw towards left', 'yaw towards right'}, ...
    {[ 0.0;  1.40;  0.0;  0.0],  ...   % up = elbow flexion upward
     [ 0.0; -0.50;  0.0;  0.0],  ...   % down = elbow extension
     [ 0.0;  0.80;  0.4;  0.0],  ...   % bicep extend = wrist pitch down
     [ 0.0;  0.70;  0.0;  1.0],  ...   % roll = wrist rotation
     [ 0.30; 0.50;  0.0;  0.5],  ...   % yaw towards left = shoulder + roll
     [-0.30; 0.50;  0.0; -0.5]}  ...   % yaw towards right = opposite
);

% Colour for each gesture
color_map = containers.Map( ...
    {'up', 'down', 'bicep extend', 'roll', ...
     'yaw towards left', 'yaw towards right'}, ...
    {[0.22 0.54 0.87], ...   % blue
     [0.85 0.35 0.19], ...   % orange
     [0.11 0.62 0.46], ...   % green
     [0.73 0.46 0.09], ...   % amber
     [0.53 0.53 0.50], ...   % gray
     [0.64 0.18 0.18]} ...   % dark red
);

%% ── LOAD PREDICTIONS CSV ────────────────────────────────────
% predictions.csv is the output of emg_classify_export.py
% It contains: predicted_label, bicep_envelope, tricep_envelope, etc.

data = readtable('predictions.csv');
fprintf('Loaded predictions.csv: %d rows\n', height(data));
fprintf('Columns: ');
disp(data.Properties.VariableNames);

% Use first 150 rows for demo
demo = data(1:min(150, height(data)), :);
n    = height(demo);

fprintf('Running simulation with %d samples\n', n);
fprintf('Unique predicted labels: ');
disp(unique(demo.predicted_label)');

%% ── BUILD SMOOTH TRAJECTORY ─────────────────────────────────
trajectory = zeros(4, n);
current    = zeros(4, 1);

for i = 1:n
    g = strtrim(demo.predicted_label{i});

    if isKey(gesture_map, g)
        target = gesture_map(g);
    else
        % Unknown label — hold current position
        fprintf('Warning: unknown label "%s" at row %d\n', g, i);
        target = current;
    end

    % Smooth interpolation — 20% toward target each frame
    current = current + 0.20 * (target - current);
    trajectory(:, i) = current;
end

%% ── SETUP FIGURE ────────────────────────────────────────────
figure('Name', 'EMG Teleoperation Simulation', ...
       'Color', 'white', ...
       'Position', [80 80 1100 650]);

joint_names = {'Shoulder', 'Elbow', 'Wrist pitch', 'Wrist roll'};
body_names  = {'upper_arm', 'forearm', 'wrist', 'end_effector'};
line_colors = {[0.22 0.54 0.87], [0.85 0.35 0.19], ...
               [0.73 0.46 0.09], [0.33 0.17 0.60]};

%% ── ANIMATE ─────────────────────────────────────────────────
fprintf('\nAnimating %d frames...\n', n);

for i = 1:n
    q       = trajectory(:, i);
    gesture = strtrim(demo.predicted_label{i});

    % Get colour for this gesture
    if isKey(color_map, gesture)
        col = color_map(gesture);
    else
        col = [0.4 0.4 0.4];
    end

    % ── Compute joint positions using forward kinematics ──────
    pts      = zeros(5, 3);
    pts(1,:) = [0 0 0];
    for k = 1:4
        T          = getTransform(robot, q, body_names{k});
        pts(k+1,:) = T(1:3, 4)';
    end
    xs = pts(:,1);
    ys = pts(:,2);
    zs = pts(:,3);

    % ── Robot arm panel ───────────────────────────────────────
    subplot(1, 2, 1);
    cla;

    % Draw arm links
    plot3(xs, ys, zs, '-', 'Color', col, 'LineWidth', 8);
    hold on;

    % Draw joint circles
    plot3(xs(2:end-1), ys(2:end-1), zs(2:end-1), 'o', ...
          'MarkerSize', 14, 'MarkerFaceColor', 'white', ...
          'MarkerEdgeColor', col, 'LineWidth', 2.5);

    % Base
    plot3(0, 0, 0, 's', 'MarkerSize', 18, ...
          'MarkerFaceColor', [0.27 0.27 0.25], 'MarkerEdgeColor', 'none');

    % End effector
    plot3(xs(end), ys(end), zs(end), 'o', 'MarkerSize', 16, ...
          'MarkerFaceColor', col, 'MarkerEdgeColor', 'white', 'LineWidth', 2);

    hold off;
    xlabel('X (m)', 'FontSize', 9);
    ylabel('Y (m)', 'FontSize', 9);
    zlabel('Z (m)', 'FontSize', 9);
    xlim([-0.40 0.40]);
    ylim([-0.40 0.40]);
    zlim([-0.05 0.45]);
    view(45, 22);
    grid on;
    set(gca, 'Color', [0.96 0.96 0.96]);
    title(sprintf('Movement:  %s', gesture), ...
          'FontSize', 14, 'FontWeight', 'bold', 'Color', col);

    % Show confidence if column exists
    if ismember('confidence', demo.Properties.VariableNames)
        conf = demo.confidence(i);
        subtitle(sprintf('Confidence: %.0f%%', conf*100), 'FontSize', 10);
    end

    % ── Joint angle traces panel ──────────────────────────────
    subplot(1, 2, 2);
    cla;
    hold on;
    t_range = max(1, i-60):i;
    for j = 1:4
        plot(t_range, rad2deg(trajectory(j, t_range)), ...
             'LineWidth', 2.2, 'Color', line_colors{j}, ...
             'DisplayName', joint_names{j});
    end
    plot([i i], [-100 100], 'r--', 'LineWidth', 1.2, ...
         'HandleVisibility', 'off');
    hold off;
    legend('Location', 'northwest', 'FontSize', 9);
    xlabel('Sample', 'FontSize', 9);
    ylabel('Angle (degrees)', 'FontSize', 9);
    title('Joint angles over time', 'FontSize', 11, 'FontWeight', 'bold');
    ylim([-100 100]);
    xlim([max(1, i-60) max(i+5, 10)]);
    grid on;
    set(gca, 'Color', [0.98 0.98 0.98]);

    drawnow;
    pause(0.08);

    fprintf('[%3d/%d]  %-22s  Elbow: %5.1f deg   Roll: %5.1f deg\n', ...
            i, n, gesture, rad2deg(q(2)), rad2deg(q(4)));
end

fprintf('\nSimulation complete.\n');