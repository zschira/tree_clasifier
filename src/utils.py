from shapely.geometry import Polygon

def compute_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def bb_2_polygons(left, bottom, boxes):
    polys = []
    
    for box in boxes:
        # Convert from ratio to meters
        box *= 20

        box[0] += left
        box[1] += bottom
        box[2] += left
        box[3] += bottom

        polys.append(Polygon([(box[0], box[3]), (box[0], box[1]), (box[2], box[1]), (box[2], box[3])]))

    return polys

def convert_predictions(predictions):
    bounds_pred = predictions[0]
    labels_pred = predictions[1].flatten()

    for (i, bounds) in enumerate(bounds_pred):
        row, col = divmod(i)
        window_left = (col * 5) / 200
        window_top = 1 - ((row * 5) / 200)
        window_centroid = [window_left + 0.1, window_top - 0.1]
        bounds[:, 0] += window_centroid[0]
        bounds[:, 1] += window_centroid[1]
        bounds[:, 2] += window_centroid[0]
        bounds[:, 3] += window_centroid[1]

    bounds_pred = np.reshape(bounds_pred, (bounds_pred.shape[0] * 9, 4))

    sort_inds = np.argsort(labels_pred)
    labels_pred = labels_pred[sort_inds]
    bounds_pred = bounds_pred[sort_inds, :]

    labels_pred[labels_pred > 0.75] = 1
    labels_pred[labels_pred < 0.75] = 0
    labels_pred = labels_pred.astype(int)

    return (bounds_pred, labels_pred)

def score_predictions(bounds_pred, labels_pred, test_dir, fname):
    if not os.path.isfile(test_dir / "labels" / (fname + ".npy")):
        return

    bounds_truth = np.load(test_dir / "bounds" / (fname + ".npy"))
    labels_truth = np.load(test_dir / "labels" / (fname + ".npy"))

    iou_avg = 0
    num_bb = 0

    print(labels_pred)

    for (b_t, l_t, b_p, l_p) in zip(bounds_truth, labels_truth, bounds_pred, labels_pred):
        if l_t != l_p:
            print("Mismatched label for file {}".format(fname))
            continue

        if l_p == 0:
            continue

        iou_avg += compute_iou(b_t, b_p)
        num_bb += 1

    print("Average iou for file, {}: {}".format(fname, iou_avg / num_bb))
