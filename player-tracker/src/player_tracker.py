import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load video
cap = cv.VideoCapture("../../data/vide0-2.mp4")

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

def init_tracker(select_frame, window_name):
    roi = cv.selectROI(window_name, select_frame, False, False)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None, None

    track_window = (x, y, w, h)
    gray = cv.cvtColor(select_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255

    points = cv.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask
    )

    return track_window, points


# Let user select the player
window_name = "Player Tracking"
track_window, points = init_tracker(frame, window_name)
if track_window is None:
    print("ROI selection canceled")
    cap.release()
    cv.destroyAllWindows()
    exit()

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
)

min_points = 15
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
last_center = None


def draw_3d_overlay(frame, trajectory):
    if len(trajectory) < 2:
        return

    h, w = frame.shape[:2]
    overlay_w = min(300, w // 3)
    overlay_h = min(200, h // 3)
    ox = w - overlay_w - 20
    oy = 20

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    zs = [p[2] for p in trajectory]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    def norm(val, vmin, vmax):
        if vmax == vmin:
            return 0.0
        return (val - vmin) / (vmax - vmin)

    z_scale_x = overlay_w * 0.3
    z_scale_y = overlay_h * 0.3

    points = []
    for x, y, z in trajectory:
        nx = norm(x, min_x, max_x)
        ny = norm(y, min_y, max_y)
        nz = norm(z, min_z, max_z)

        px = int(ox + nx * overlay_w + nz * z_scale_x)
        py = int(oy + (1 - ny) * overlay_h - nz * z_scale_y)
        points.append((px, py))

    cv.rectangle(frame, (ox, oy), (ox + overlay_w, oy + overlay_h), (60, 60, 60), 1)
    cv.putText(
        frame,
        "3D path (proj)",
        (ox, max(10, oy - 5)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    for i in range(1, len(points)):
        cv.line(frame, points[i - 1], points[i], (255, 255, 0), 2)


def smooth_trajectory(traj, window=5):
    if len(traj) < window:
        return traj

    smoothed = []
    for i in range(len(traj)):
        xs, ys = [], []
        for j in range(max(0, i - window), i + 1):
            xs.append(traj[j][0])
            ys.append(traj[j][1])
        smoothed.append((int(np.mean(xs)), int(np.mean(ys)), traj[i][2]))
    return smoothed

# Trajectory list (x, y, frame_index)
trajectory = []
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if points is not None and len(points) >= min_points:
        next_points, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)
        good_new = next_points[status.flatten() == 1]
        good_old = points[status.flatten() == 1]

        if len(good_new) >= min_points:
            xs = good_new[:, 0, 0]
            ys = good_new[:, 0, 1]

            x_min = int(np.min(xs))
            y_min = int(np.min(ys))
            x_max = int(np.max(xs))
            y_max = int(np.max(ys))

            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            last_center = (center_x, center_y)

            trajectory.append((center_x, center_y, frame_index))
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            for p in good_new:
                cv.circle(frame, (int(p[0][0]), int(p[0][1])), 2, (0, 255, 255), -1)

            points = good_new.reshape(-1, 1, 2)
        else:
            points = None
    else:
        points = None

    if points is None:
        cv.putText(
            frame,
            "Tracking lost - press r to reselect",
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    smoothed_trajectory = smooth_trajectory(trajectory)

    # Draw trajectory
    for i in range(1, len(smoothed_trajectory)):
        p1 = (smoothed_trajectory[i - 1][0], smoothed_trajectory[i - 1][1])
        p2 = (smoothed_trajectory[i][0], smoothed_trajectory[i][1])
        cv.line(frame, p1, p2, (0, 0, 255), 2)

    draw_3d_overlay(frame, smoothed_trajectory)

    cv.imshow(window_name, frame)

    key = cv.waitKey(30) & 0xFF
    if key == ord('r'):
        new_track_window, new_points = init_tracker(frame, window_name)
        if new_track_window is not None:
            track_window, points = new_track_window, new_points
            trajectory = []
            frame_index = 0
            last_center = None
    elif key == 27:
        break

    prev_gray = gray
    frame_index += 1

cap.release()
cv.destroyAllWindows()

smoothed_trajectory = smooth_trajectory(trajectory)

if smoothed_trajectory:
    xs = [p[0] for p in smoothed_trajectory]
    ys = [p[1] for p in smoothed_trajectory]
    zs = [p[2] for p in smoothed_trajectory]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Frame")
    ax.set_title("Player Trajectory (3D)")
    plt.show()
