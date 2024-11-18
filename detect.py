import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas

def threshold(spectrogram, p):
    rows = spectrogram.shape[0]
    columns = spectrogram.shape[1]
    thresholded_spectrogram = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if spectrogram[i][j] < p: #adjust threshold
                thresholded_spectrogram[i][j]  = 0.0
            else:
                thresholded_spectrogram[i][j]  = 1.0
    return thresholded_spectrogram

def calc_probability(thresholded_spectrogram):
    rows = thresholded_spectrogram.shape[0]
    columns = thresholded_spectrogram.shape[1]
    tx_time_prob = np.zeros(rows)
    for i in range(rows):
        tx_time_prob[i] = sum(thresholded_spectrogram[i])/len(thresholded_spectrogram[i])

    tx_freq_prob = np.zeros(columns)
    for i in range(columns):
        tx_freq_prob[i] = sum(thresholded_spectrogram[:,i])/len(thresholded_spectrogram[:,i])

    probability_matrix  = np.outer(tx_time_prob, tx_freq_prob)

    return tx_freq_prob, tx_time_prob, probability_matrix

def choose_initial_thresh(spectrogram):
    low = np.min(spectrogram)
    high = np.max(spectrogram)
    threshes = np.linspace(low, high, 100)
    return calc_initial_thresh(spectrogram, threshes)

def calc_max_diff(arr):
    arr = sorted(arr)

    return max(
        [abs(arr[i] - arr[i+1]) for i in range(0, len(arr)-1)]
    )

def calc_initial_thresh(spectrogram, threshes):
    max_diffs = []
    for thresh in threshes: 
        threshed = threshold(spectrogram, thresh)

        tx_freq_prob, tx_time_prob, probability_matrix = calc_probability(threshed)

        max_freq_diff = calc_max_diff(tx_freq_prob)
        max_time_diff = calc_max_diff(tx_time_prob)

        max_diffs.append((max_freq_diff, max_time_diff, thresh))

        # print(max_diffs[-1], thresh)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # plt.plot(tx_freq_prob)
        # plt.title("freq or time prob")
        # plt.xlabel("freq bins or time")
        # plt.ylabel("Tx Probability")

        # ax2 = fig.add_subplot(222)
        # plt.plot(tx_time_prob)
        # plt.title("freq or time prob")
        # plt.xlabel("freq bins or time")
        # plt.ylabel("Tx Probability")

        # ax3 = fig.add_subplot(223)
        # plt.imshow(probability_matrix, cmap=plt.cm.spring, aspect='auto')
        # plt.xlabel("Frequency [MHz]")
        # plt.ylabel("Time [ms]")

        # ax4 = fig.add_subplot(224)
        # plt.imshow(threshed, cmap=plt.cm.spring, aspect='auto')
        # plt.xlabel("Frequency [MHz]")
        # plt.ylabel("Time [ms]")
        # plt.show()


    freq_diffs = [x[0] for x in max_diffs]
    time_diffs = [x[1] for x in max_diffs]

    max_freq_diff = max(zip(freq_diffs, threshes))
    max_time_diff = max(zip(time_diffs, threshes))

    if max_freq_diff[1] > max_time_diff[1]: # choose higher thresh
        init_thresh = max_freq_diff[1]
    else:
        init_thresh = max_time_diff[1]

    return init_thresh 

# this section definitely needs some work

def choose_prob_thresh(tx_prob):

    return find_highest_thresh_of_highest_normalized_rate_sum(tx_prob)

def calc_intersections_and_sum_of_rates(vec, v):
    intersections = 0
    sum_of_abs_rates = 0

    for i, curr in enumerate(vec[:-1]):
        nextt = vec[i+1]
        if (curr <= v and v <= nextt) or (curr >= v and v >= nextt):
            diff = nextt - curr
            intersections += 1
            sum_of_abs_rates += abs(diff)

    return intersections, sum_of_abs_rates

def find_highest_thresh_of_highest_normalized_rate_sum(vec):
    threshes = np.linspace(min(vec), max(vec), 100)
    results = [calc_intersections_and_sum_of_rates(vec, thresh) for thresh in threshes]
    intersections = [x[0] for x in results]
    sum_of_abs_rates = [x[1] for x in results]
    normalized_abs_rate_sums = [x[0]/x[1] if x[1]!=0 else 0 for x in zip(sum_of_abs_rates, intersections)]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    plt.plot(vec)
    plt.title("freq or time prob")
    plt.xlabel("freq bins or time")
    plt.ylabel("Tx Probability")

    ax2 = fig.add_subplot(222)
    plt.plot(threshes, intersections)
    plt.title("intersections vs threshold")
    plt.xlabel("threshold")
    plt.ylabel("intersections")

    ax3 = fig.add_subplot(223)
    plt.plot(threshes, sum_of_abs_rates)
    plt.title("∑|(yk2-yk1)| vs threshold")
    plt.xlabel("threshold")
    plt.ylabel("∑|(yk2-yk1)|")

    ax4 = fig.add_subplot(224)
    plt.plot(threshes, normalized_abs_rate_sums)
    plt.title("∑|(yk2-yk1)| / intersections vs threshold")
    plt.xlabel("threshold")
    plt.ylabel("∑|(yk2-yk1)| / intersections")

    plt.show()

    thresh = max(zip(threshes, normalized_abs_rate_sums), key=lambda x: (x[1], x[0])) [0]
    # print(thresh)

    return thresh

def remove_detection(spectrogram, detection, nf_mean=None, nf_std_dev=None):
    # I should probably try to adopt some sort of masking like SCAN did on subsequent iterations
    # but rn this function requires knowledge of the noise floor's distrobution

    # otherwise it just uses the minimum value found in the spectrogram to 'remove' the previous detection
    # but that leads to *major* issues, namely detect will incorrectly characterize everything but the previous detection as
    # transmission bc it treats the previous detection (now replaced by the minimum) as the new noise floor
    # and everything else as transmission.

    if not (nf_mean is None or nf_std_dev is None):
        gen = np.random.default_rng()
        vals = iter(gen.normal(nf_mean, nf_std_dev, (int(np.sum(detection)))))
    else:
        def gen(val):
            while True:
                yield val
        vals = gen(np.min(spectrogram))

    rows, cols = detection.shape
    out = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if detection[i, j]:
                out[i, j] = next(vals)
            else:
                out[i, j] = spectrogram[i, j]
    return out

def stopping_condition():
    # I dont have a stopping condition yet so detect just asks the user if they want to continue
    return input("cont? [y/n]").lower() != "y"

def detect(spectrogram, nf_mean=None, nf_std_dev=None):
    goods = []

    while True:
        if stopping_condition():
            break
    
        p = choose_initial_thresh(spectrogram)
        thresholded_spectrogram = threshold(spectrogram, p)
        tx_freq_prob, tx_time_prob, probability_matrix = calc_probability(thresholded_spectrogram)
        freq_prob_thresh = choose_prob_thresh(tx_freq_prob)
        time_prob_thresh = choose_prob_thresh(tx_time_prob)
        prob_thresh = freq_prob_thresh * time_prob_thresh
        thresholded_prob_matrix = threshold(probability_matrix, prob_thresh)

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        plt.imshow(spectrogram, cmap=plt.cm.spring, aspect='auto')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [ms]")
        
        ax2 = fig.add_subplot(222)
        plt.plot(tx_freq_prob)
        plt.axhline(y=freq_prob_thresh, color='r', linestyle='-')
        plt.title("freq prob")
        plt.xlabel("freq bins")
        plt.ylabel("Tx Probability")
        
        ax3 = fig.add_subplot(223)
        plt.plot(tx_time_prob)
        plt.axhline(y=time_prob_thresh, color='r', linestyle='-')
        plt.title("time prob")
        plt.xlabel("Time [ms]")
        plt.ylabel("Tx Probability")
        
        ax4 = fig.add_subplot(224)
        plt.imshow(thresholded_prob_matrix, cmap=plt.cm.spring, aspect='auto')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [ms]")
        plt.show()

        # prev_detection = thresholded_prob_matrix

        if input("should we keep this iteration? [y/n]: ").lower() == "y":
            goods.append(thresholded_prob_matrix)
        spectrogram = remove_detection(spectrogram, thresholded_prob_matrix, nf_mean, nf_std_dev)
    
    summ = np.zeros(goods[0].shape)
    for good in goods:
        summ += good

    return summ


def read_spectrogram(testfile, groundtruthfile):
    spectrogram = pandas.read_csv(testfile, header=None)
    spectrogram = spectrogram.to_numpy()

    groundtruth = pandas.read_csv(groundtruthfile, header=None)
    groundtruth = groundtruth.to_numpy()

    return spectrogram, groundtruth

def jaccard_similarity(detection, groundtruth):
    truePositives = 0 #ground truth and algorithm both detect transmission
    trueNegatives = 0 #ground truth and algorithm both do not detect transmission
    falsePositives = 0 #ground truth says no transmission, algorithm says transmission
    falseNegatives = 0 #ground truth says transmission, algorithm says no transmission

    rows = detection.shape[0]
    columns = detection.shape[1]
    
    for row in range(rows):
        for column in range(columns):
            if groundtruth[row][column] == detection[row][column] == 1:
                truePositives +=1
            elif groundtruth[row][column] == detection[row][column] == 0:
                trueNegatives +=1
            elif detection[row][column] == 1:
                falsePositives +=1
            else:
                falseNegatives += 1
    
    #print(truePositives, trueNegatives, falsePositives, falseNegatives)
    
    jac = truePositives/(truePositives + falsePositives + falseNegatives)
    
    return jac


def main():
    prefix = "C:\\Users\\Clark\\Documents\\ubinetlab\\tx_detection\\"
    prefix += "testFilesEasyToHard\\"
    trace = "hard"
    # trace = "Hydroxl3_gain38"
    testfile = prefix + trace + ".csv"
    groundtruthfile = prefix + trace + "_groundtruth.csv"
    spectrogram = pandas.read_csv(testfile, header=None)
    spectrogram = spectrogram.to_numpy()

    gt = pandas.read_csv(groundtruthfile, header=None)
    gt = gt.to_numpy()

    fft_size = 1024
    sample_rate = 1e6    

    # print("here")
    detection = detect(spectrogram)

    print(jaccard_similarity(detection, gt))


    # main

    prefix = "C:\\Users\\Clark\\Documents\\ubinetlab\\tx_detection\\"

    # snrs = [-106, -105, -104, -102, -100, -95, -90]
    # snrs = [-102]
    # snr_dict = {snr:list() for snr in snrs}

    # for snr in snrs:
    #     for idx in range(1,101):
    #         testprefix = prefix + f"vary_snr\\snr_{snr}\\"
    #         groundtruthprefix = prefix + f"vary_snr\\groundtruth\\snr_{snr}\\"
    #         trace = f"test_file_{snr}_2_{snr}_{idx}_"
    #         testfile = testprefix + trace + ".csv"
    #         groundtruthfile = groundtruthprefix + trace + "groundtruth_.csv"

    #         spectrogram, groundtruth = read_spectrogram(testfile, groundtruthfile)
    #         detection = detect(spectrogram)
    #         j = jaccard_similarity(detection, groundtruth)
    #         print(idx, j)

    #         snr_dict[snr].append(j)

    # fig, ax = plt.subplots()
    # ax.boxplot(snr_dict.values())
    # ax.set_xticklabels(snr_dict.keys())
    # plt.show()

if __name__ == "__main__":
    main()