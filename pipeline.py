# -*- coding: utf-8 -*-
"""
Case Studies in Data Analytics-Assignment 1

Author: Saikrishna Javvadi

Main script for running the complete object detection and classification pipeline.

References:  https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
             https://github.com/qqwweee/keras-yolo3
             https://pjreddie.com/darknet/yolo/
             https://melvinkoh.me/solving-producerconsumer-problem-of-concurrent-programming-in-python-ck3bqyj1j00i8o4s1cqu9mfi7
             https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
"""

import sys
sys.path.append('keras-yolo3-master')

from multiprocessing import Process, Queue
from PIL import Image, ImageFont, ImageDraw
from yolo import YOLO
from tensorflow import keras
import cv2
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FRAME_RATE = 30  # frames per second

# Allowing user to choose a  query to run if required
DO_ANALYSIS = False
QUERY = 'Query2'
try:
    QUERY = sys.argv[1]
except:
    QUERY = 'Query2'
if QUERY == 'Query2':
    DO_ANALYSIS = True


results = {}  # initialising an empty dictionary to store the query results from each frame

default_result_row = {'Sedan': 0,
                      'SUV': 0,
                      'Total': 0}

def stream_video(frame_queue):
    print("Starting Streaming Video")
    # Read until video is completed
    video_capture = cv2.VideoCapture('./video.mp4')
    while (video_capture.isOpened()):
        # capturing frame by frame
        ret, frame = video_capture.read()
        if ret == True:
            # Adding the frame to queue
            frame_queue.put(frame)

            # Delay by frame rate
            time.sleep(1 / FRAME_RATE)

            # Break the loop if "Q" is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object after all the frames are streamed
    video_capture.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def frame_processor(frame_queue):
    print("Starting Processing frames")
    global QUERY
    global DO_ANALYSIS

    frame_num = 0

    # Instantiating YOLO , using YOLO-V3 weights
    yolo = YOLO(model_path='./model_data/yolo_v3.h5',
                anchors_path='./model_data/yolo_anchors.txt',
                classes_path='./model_data/coco_classes.txt')

    # Instantiate car type classifier
    model = keras.models.load_model('./mobilenet_cars.h5')

    # Font for annotations
    font1 = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=11)
    font2 = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=15)

    # Arraylist to store the query statistics
    query1_stats = []
    query2_stats = []

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_video/Saikrishna_Javvadi_Assignment1_video.mp4', fourcc, 30, (360, 288))

    pipeline_start_time = time.time()

    while True:
        # retrieve the frame from the queue
        frame = frame_queue.get()

        start_time = time.time()
        frame_num += 1
        print(f'\nFrame {frame_num}')
        video_time = frame_num * (1 / 25)  # 25 is the true frame rate of the video
        print(f'Video Time {round(video_time, 2)} s')
        new_result = copy.deepcopy(default_result_row)

        if QUERY in ['Query1', 'Query2']:
            # ------------------------ STEP 1 ------------------------
            # Using YOLO to detect car objects
            car_boxes, dims_car_boxes = get_car_bounding_boxes(frame, yolo)
            new_result['Total'] = len(car_boxes)  # Total number of cars in the given frame
            query1_stats.append([frame_num, time.time() - start_time])
            # ---------------------------------------------------------
        if QUERY in ['Query2', 'Query3']:
            # ------------------------ STEP 2 ------------------------
            car_types = []
            for car_box in car_boxes:
                # Using MobileNet to classify SUV or Sedan
                car_type = get_car_type(car_box, model)
                car_types.append(car_type)
            if len(car_boxes) > 0:
                query2_stats.append([frame_num, time.time() - start_time])

            image = Image.fromarray(frame)
            # Create annotations for video frame
            for dims_car_box, car_type in zip(dims_car_boxes, car_types):
                draw = ImageDraw.Draw(image)
                if car_type == 'Sedan':
                    box_colour = (0, 0, 255)  # Blue
                else:
                    box_colour = (255, 0, 0)  # Red
                for i in range(3):
                    # Drawing a bounding box around car
                    draw.rectangle([dims_car_box['left'] + i, dims_car_box['top'] + i, dims_car_box['right'] - i,
                                    dims_car_box['bottom'] - i], outline=box_colour)
                # Adding a label to top left of the bounding box
                label = f"{car_type}"
                label_width, label_height = draw.textsize(label, font1)
                draw.rectangle([dims_car_box['left'], dims_car_box['top'], dims_car_box['left'] + label_width,
                                dims_car_box['top'] + label_height], fill=box_colour)
                draw.text((dims_car_box['left'], dims_car_box['top']), label, font=font1, fill=(0, 0, 0, 128))

            # Add label to top left with count of cars
            draw = ImageDraw.Draw(image)
            label = f"Car Count: {len(car_boxes)}"
            label_width, label_height = draw.textsize(label, font2)
            draw.rectangle([0, 0, label_width, label_height], fill=(0, 0, 0))  # Black Rectangle
            draw.text((0, 0), label, font=font2, fill=(255, 255, 255, 128))  # White text
            annotated_frame = np.asarray(image)
            cv2.imshow("Object Detection", annotated_frame)

            # Write video
            out.write(annotated_frame)
            # ---------------------------------------------------------
        process_time = time.time() - start_time
        print(f'{round(process_time, 3)} s to process ...')

        # Info message
        if QUERY == 'Query1':
            for _ in car_boxes:
                print(f'Query1 : Car detected')
        elif QUERY == 'Query2':
            for car_type in car_types:
                print(f'Query2 : {car_type} detected')

                new_result[car_type] += 1

            results[frame_num] = copy.deepcopy(new_result)

            # Write results to file
            csv_string = f"\n{frame_num}, {new_result['Sedan']}, " + \
                         f"{new_result['SUV']}, {new_result['Total']}"
            with open('./results.csv', 'a') as f:
                f.write(csv_string)

        # Break the loop if "Q" is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # Or if the queue is empty
        elif frame_queue.empty():
            break
        elif cv2.waitKey(25) & 0xFF == ord('1'):
            QUERY = 'Query1'
            DO_ANALYSIS = False
        elif cv2.waitKey(25) & 0xFF == ord('2'):
            QUERY = 'Query2'
            DO_ANALYSIS = False
    throughput = time.time() - pipeline_start_time
    with open('./output.txt', 'w') as f:
        out_string = f'\nThroughput : {np.round(throughput, 4)} s\n'
        print(out_string)
        f.write(out_string)
    # Save event extraction times
    if QUERY == 'Query1':
        query1_stats = np.array(query1_stats)
        np.save("./query1_stats.npy", query1_stats)
    elif QUERY == 'Query2':
        query1_stats = np.array(query1_stats)
        query2_stats = np.array(query2_stats)
        np.save("./query1_stats.npy", query1_stats)
        np.save("./query2_stats.npy", query2_stats)

    out.release()


def get_car_bounding_boxes(frame, yolo):
    # Convert to an image
    image = Image.fromarray(frame)

    # Using tiny YOLO to detect objects
    r_image, object_boxes, time_value = yolo.detect_image(image)

    # Iterate through all the objects detected in the frame
    car_boxes = []
    car_box_dims = []
    for object_box in object_boxes:
        # If the object detected is of type car
        if object_box['class'] == 'car':
            car_box_dims.append(object_box)

            # Storing the bounding box dimensions
            top = object_box['top']
            left = object_box['left']
            bottom = object_box['bottom']
            right = object_box['right']

            # get the bounding box of the object in the frame
            car_box = frame[top:bottom, left:right]  # Pass this to next stage of the pipeline
            car_boxes.append(car_box)

    return car_boxes, car_box_dims


def get_car_type(car_box, model):
    new_car_box = Image.fromarray(car_box)
    new_car_box = new_car_box.resize((160, 160), Image.ANTIALIAS)
    result = np.asarray(new_car_box)
    result = np.expand_dims(result, axis=0)
    images = np.vstack([result])
    classes = model.predict(images, batch_size=10)
    print(classes)
    if classes[0] > 0.55:  # Classification threshold
        car_type = 'Sedan'
    else:
        car_type = 'SUV'
    return car_type

#Reference: https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/

def compute_results():
    names = ['Frame Number', 'Sedan', 'SUV', 'Total']
    ground_truth = pd.read_excel("./Groundtruth.xlsx", index_col=0, names=names)

    predictions = pd.read_csv("./results.csv", skiprows=1, index_col=0, names=names)

    f1_scores = []
    # Query1
    y_true = ground_truth.loc[:, names[-1]].values
    y_pred = predictions.loc[:, names[-1]].values

    true_pos, false_pos, false_neg = get_tp_fp_fn(y_true, y_pred)

    # Calculate F1-score
    f1_scores.append(compute_f1score(true_pos, false_pos, false_neg))

    # Query2
    true_pos, false_pos, false_neg = 0, 0, 0
    true_values = ground_truth[ground_truth['Total'].values != 0]
    pred_values = predictions[ground_truth['Total'].values != 0]

    for cols in (names[1], names[2]):
        y_true = pd.DataFrame(true_values.loc[:, cols]).sum(axis=1).values
        y_pred = pd.DataFrame(pred_values.loc[:, cols]).sum(axis=1).values

        tp, fp, fn = get_tp_fp_fn(y_true, y_pred)
        true_pos += tp
        false_pos += fp
        false_neg += fn

    # Calculate F1-score
    f1_scores.append(compute_f1score(true_pos, false_pos, false_neg))

    plt.bar([f"Q{i}" for i in range(1, 3)], f1_scores)
    plt.title('Query F1 Scores')
    plt.grid(axis='y', alpha=0.5)
    plt.ylim([0, 1])
    plt.show()

    # Print scores
    with open('./output.txt', 'a') as f:
        f.write('\nF1 Scores\n')
        f.write('---------\n')
        for i, f1 in enumerate(f1_scores):
            f.write(f'Q{i + 1} : {round(f1, 3)}\n')


def compute_f1score(true_pos, false_pos, false_neg):
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return 2 * (precision * recall) / (precision + recall)


def get_tp_fp_fn(y_true, y_pred):
    true_pos, false_pos, false_neg = 0, 0, 0
    for i in range(len(y_true)):
        # Calculating True Positive's, False Positive's and False Negative's
        if y_pred[i] <= y_true[i]:
            true_pos += y_pred[i]
            false_pos += 0
            false_neg += y_true[i] - y_pred[i]
        else:
            true_pos += y_true[i]
            false_pos += y_pred[i] - y_true[i]
            false_neg += 0
    return true_pos, false_pos, false_neg


def make_piecewise(stats):
    N = stats.shape[0]
    previous = stats[0, 0]
    j = 1
    for _ in range(1, N):
        current = stats[j, 0]
        if current != previous + 1:
            # Insert NaN for frames with gaps that doesn't have any data
            stats = np.insert(stats, j, np.array([np.nan, np.nan]), axis=0)
            j += 1
        previous = current
        j += 1
    return stats


if __name__ == '__main__':
    # Create CSV file to store results
    with open('./results.csv', 'w') as f:
        csv_header = 'Frame Number, Sedan, SUV, Total'
        f.write(csv_header)

    # Creating a queue for storing video frames
    frame_queue = Queue()

    # creating two processes
    process1 = Process(target=frame_processor, args=(frame_queue,))
    process2 = Process(target=stream_video, args=(frame_queue,))

    # starting the processes
    process1.start()
    process2.start()

    # waiting for the started processes to complete
    process1.join()
    process2.join()

    if QUERY == 'Query2' and DO_ANALYSIS == True:
        # Compute F1-score for each query
        compute_results()

        # Plotting
        query1_stats = np.load("./query1_stats.npy")
        query2_stats = np.load("./query2_stats.npy")

        with open('./output.txt', 'a') as f:
            f.write('\nAverage extraction times\n')
            f.write('------------------------\n')
            for i, q_stats in enumerate([query1_stats, query2_stats]):
                average_extraction = np.round(np.mean(q_stats[:, 1]), 4)
                std_extraction = np.round(np.std(q_stats[:, 1]), 4)
                f.write(f'Q{i + 1} : {average_extraction} +- {std_extraction} s\n')

        query1_stats = make_piecewise(query1_stats)
        query2_stats = make_piecewise(query2_stats)

        plt.plot(query1_stats[:, 0], query1_stats[:, 1], 'g', alpha=1, linewidth=1)
        plt.plot(query2_stats[:, 0], query2_stats[:, 1], 'b', alpha=0.7, linewidth=1)
        plt.legend(['Query1', 'Query2'])
        plt.title('Query Extraction Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Time (seconds)')
        plt.show()
