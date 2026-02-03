import cv2
import time

def fourcc_to_str(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def probe_camera_force(
    camera_index=1,
    backend=cv2.CAP_DSHOW,
    req_w=3200,
    req_h=1200,
    req_fps=60,
    req_fourcc="MJPG",
    probe_seconds=5.0,
    warmup_seconds=0.7,
):
    print("=== Camera Probe (Forced Mode) ===")
    print(f"camera_index = {camera_index}")
    print(f"backend      = {backend}")
    print(f"request      = {req_w}x{req_h} @ {req_fps}fps, FOURCC={req_fourcc}")

    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    # Force codec (critical for your camera to reach 3200x1200@60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*req_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)

    # Read one frame to finalize negotiation
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Cannot read frame after setting properties")
        cap.release()
        return

    H, W = frame.shape[:2]
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    got_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    print(f"[NEGOTIATED] size   = {W} x {H}")
    print(f"[NEGOTIATED] fps    = {got_fps} (often 0.0 on some drivers; measured fps below is more reliable)")
    print(f"[NEGOTIATED] fourcc = {got_fourcc}")

    print("[INFO] Probing... press 'q' to quit")

    t0 = time.perf_counter()
    n_total = 0
    n_measured = 0
    t_measure_start = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed")
            break

        n_total += 1
        now = time.perf_counter()
        elapsed = now - t0

        # Warmup: ignore initial frames for FPS measurement
        if elapsed >= warmup_seconds:
            if t_measure_start is None:
                t_measure_start = now
            n_measured += 1

        # Draw split line
        mid_x = frame.shape[1] // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]-1), (0, 255, 0), 2)

        # Overlay
        cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]} total={n_total}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Camera Probe Forced (press q)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if elapsed >= probe_seconds:
            break

    if t_measure_start is not None:
        t_end = time.perf_counter()
        measured_fps = n_measured / max(t_end - t_measure_start, 1e-6)
        print(f"[MEASURED] FPS ~ {measured_fps:.2f}  (after {warmup_seconds:.1f}s warmup)")
    else:
        print("[MEASURED] Not enough time to measure FPS (warmup too long or too short probe)")

    cap.release()
    cv2.destroyAllWindows()
    print("=== Probe Finished ===")


if __name__ == "__main__":
    # 你已经确认 FFmpeg 列表支持 3200x1200 mjpeg 60fps，所以这里直接按该模式 probe
    probe_camera_force(
        camera_index=1,
        backend=cv2.CAP_DSHOW,
        req_w=3200,
        req_h=1200,
        req_fps=60,
        req_fourcc="MJPG",
        probe_seconds=5.0,
    )
