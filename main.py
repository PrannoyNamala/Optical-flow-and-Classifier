import cv2
import numpy as np
import matplotlib.pyplot as plt


def lucas_kanade_method(video_path="Cars On Highway.mp4"):
    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), )

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    op_shape = (old_frame.shape[1], old_frame.shape[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('sparselk.avi', fourcc, 25.0, op_shape)

    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # Display the demo
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        out.write(img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


def dense_optical_flow():
    # Read the video
    cap = cv2.VideoCapture("Cars On Highway.mp4")
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_hsv = cv2.VideoWriter('denselkhsv.avi', fourcc, 25.0, (1920, 1080))
    out_quiver = cv2.VideoWriter('denselkquiver.avi', fourcc, 25.0, (640, 480))
    while True:
        plt.figure()
        _, new_frame = cap.read()
        if not _:
            cap.release()
            cv2.destroyAllWindows()
            break
        frame_copy = new_frame
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        # print(new_frame.shape)
        flow = cv2.optflow.calcOpticalFlowSparseToDense(old_frame, new_frame, grid_step=25, k=128, sigma=0.05)
        # compare = np.hstack((flow[:,:,0], flow[:,:,1]))
        # plt.imshow(compare, cmap='gray')
        # plt.show()
        # raise SystemExit()
        point_x = []
        point_y = []
        mag_x = []
        mag_y = []
        i = 0
        for row in flow:
            j = 0
            for element in row:
                if i % 25 == 0 and j % 25 == 0:
                    point_x.append(1079 - i)
                    point_y.append(j)
                    mag_x.append(element[0])
                    mag_y.append(element[1])
                j += 1
            i += 1
        plt.quiver(point_x, point_y, mag_x, mag_y)
        plt.savefig("load.png")
        plt.close('all')
        read = cv2.imread("load.png")
        out_quiver.write(read)
        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        out_hsv.write(bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame
        old_frame = new_frame


def background_eliminator():
    cap = cv2.VideoCapture('Cars On Highway.mp4')

    fgbg = cv2.createBackgroundSubtractorKNN()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('no_bg.avi', fourcc, 25.0, (1920, 1080))

    while (1):
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.threshold(fgmask, 130, 255, cv2.THRESH_BINARY)

        new_frame = cv2.bitwise_and(frame, frame, mask=fgmask[-1])

        cv2.imshow('Masked Frame', new_frame)
        out.write(new_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lucas_kanade_method()
    print("Sparse LK done")
    dense_optical_flow()
    print("Dense LK done")
    background_eliminator()
    raise SystemExit()
