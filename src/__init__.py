import face_detection as fd
import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt



def train():
    fd.training_data_collection()


# distance function to calculate the distance between two points
# in euclidean space having n dimensions(30000 in this case)
def distance(p1, p2):
    return np.sum((p2 - p1) ** 2) ** .5



# knn function to find the k nearest points to the input point
# first we will calc distance from input to each training data point then sort(obv in increasing order) them and take
# first k and make predictions based upon them

def knn(X, Y, test, names, k = 10):
    # it will have the distance( from the input point ) and label of each point as tuple ie : (distance, label)
    d = []
    r = X.shape[0]
    for i in range(r):
        d.append((distance(test, X[i]), Y[i]))

    # l is the list of sorted distance label
    l = np.array(sorted(d))[:, 1]
    l = l[:k]
    u = np.unique(l, return_counts=True)

    # convert the unique labels with their frequency into key value pair
    freq_dict = dict()
    for i in range(len(u[0])):
        freq_dict[u[0][i]] = u[1][i]

    # get the key whose value is maxmimum in the dictionary
    pred = int(max(freq_dict, key=freq_dict.get))

    accuracy = int(freq_dict[pred])

    percentage_accuracy = int((accuracy / k) * 100)

    result = names[pred]
    return result


def face_recognition():
    dataset_path = "./facedata/"
    labels = []
    class_id = 0
    names = {}
    face_data = []
    labels = []

    for fx in os.listdir(dataset_path):
        if fx.endswith(".npy"):
            names[class_id] = fx[:-4]
            print("Loading file ", fx)
            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)

            # Create Labels
            target = class_id * np.ones((data_item.shape[0],))
            labels.append(target)
            class_id += 1

    X = np.concatenate(face_data, axis=0)
    Y = np.concatenate(labels, axis=0)

    # print(X.shape)
    # print(Y.shape)

    # get the camera object
    cam = cv2.VideoCapture(0)
    # video stream
    while True:
        ret, frame = cam.read()

        # if camera resource is not available wait and check again
        if not ret:
            continue

        # q is pressed on frame then quit the window
        key_pressed = cv2.waitKey(1) & 0xFF  # bit masking to get last 8-bits
        if key_pressed == ord('q'):
            break;

        face_section = fd.detect(frame)
        test = cv2.resize(face_section, (100, 100))
        test = np.array(test)
        test = test.flatten()

        #print(face_section.shape)

        # see the prediction on console
        result = knn(X, Y, test, names)
        print(result)

        # Write the prediction label on the fram
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (280, 80)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(frame, result,
            position,
            font,
            fontScale,
            fontColor,
            lineType,
            cv2.LINE_AA
        )

        # now show the frame
        cv2.imshow("Video Frame", frame)

    # release the camera after quitting the window
    cam.release()
    cv2.destroyAllWindows()
    return



if __name__ == "__main__":
    print("1. press 1 to train model\n2. press 2 to test model\n3. press 0 to quit !\n")
    choice = int(input())
    if choice == 1:
        train()
    elif choice == 2:
        face_recognition()
    else:
        print("\n-----Thank you --------\n")


