import argparse
import ast
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

feature_cols = [
    "duration",
    "packet_rate",
    "byte_rate",
    "avg_pkt_size",
    "syn_ratio",
    "fin_ratio",
    "rst_ratio",
    "flag_entropy",
    "ttl_entropy",
    "ipid_entropy",
    "session_coherence"
]

# Cost matrix: cost[action][true_label]
COST_2_ACTION = {
    0: {0: 0.0,   1: 1},   # allow
    1: {0: 0.4,   1: 0.2}  # normalize
}


COST_3_ACTION = {
    0: {0: 0.0,   1: 1.0},   # allow
    1: {0: 0.2,   1: 0.4},   # capture
    2: {0: 0.4,   1: 0.2}  # normalize
}

def compute_threshold(a, b, cost):
    """
    Compute decision threshold between two actions using cost analysis.
    
    The threshold P* is where the expected cost is equal:

    Parameters:
        a, b: action indices
        cost: cost matrix
    
    Returns:
        float or None: probability threshold for choosing action b over a
    """
    num = cost[a][0] - cost[b][0]
    den = (cost[b][1] - cost[b][0]) - (cost[a][1] - cost[a][0])
    return num / den if den != 0 else None

def naive_warden(p_cal):
    """
    Random action selection based on covert probability.
    
    Takes action 1 (block/normalize) with probability p_cal[i].
    This is a baseline for comparison.
    
    Parameters:
        p_cal: array of calibrated probabilities
    
    Returns:
        array of actions (0 or 1)
    """
    rng = np.random.default_rng()
    actions=np.array([rng.random() < p_cal[i] for i in range(len(p_cal))],dtype=int)
    return actions

def parse_flow_id(flow_id):
    """
    Parse flow_id from tuple string format.
    
    Format: "((src_ip, src_port), (dst_ip, dst_port), proto)"
    
    Parameters:
        flow_id: string representation of flow tuple
    
    Returns:
        tuple: (src_ip, src_port, dst_ip, dst_port, proto)
    """
    ((src_ip, src_port), (dst_ip, dst_port), proto) = ast.literal_eval(flow_id)
    return src_ip, src_port, dst_ip, dst_port, proto

def eval_traffic_class(flow_id):
    """
    Create traffic class E = (src_ip, protocol, destination_port).
    
    Traffic class is a tuple of source IP, transport protocol, and destination port.
    Used to group flows for per-class decision making.
    
    Parameters:
        flow_id: string representation of flow tuple
    
    Returns:
        tuple: (src_ip, protocol, dst_port)
    """
    src_ip, src_port, dst_ip, dst_port, proto = parse_flow_id(flow_id)
    return (src_ip, proto, dst_port)

def per_class_naive_warden(eval_data,p_cal,time_window=1000.0,lambda_param=0.5, seed=42):
    """
    Applies actions per traffic class within time windows on covert probability.
    
    Takes action 1 (block/normalize) with mean probability p_{E,cal}[i] for a traffic class E.
    This is a baseline for comparison.
    
    Parameters:
        eval_data: pandas DataFrame with "flow_id", "end_time"
        p_cal: array of calibrated probabilities
        time_window: duration of time window (default: 1000)
        lambda_param: smoothing factor (0, 1] (default: 0.5)
        seed: random seed for reproducibility (default: 42)
    
    Returns:
        array of actions (0 or 1)
    """
    # Initialize results
    actions = np.zeros(len(eval_data), dtype=int)
    
    # Extract traffic classes
    eval_data_copy = eval_data.copy()
    eval_data_copy["traffic_class"] = eval_data_copy["flow_id"].apply(eval_traffic_class)
    eval_data_copy["p_cal"] = p_cal
    
    # Dictionary to store previous smoothed probability for each traffic class
    # p_{t-1}(E) for each class E
    previous_probs = {}
    
    # Get time boundaries
    begin_time = eval_data_copy["end_time"].min()
    end_time = eval_data_copy["end_time"].max()
    
    actual_time = begin_time
    
    rng = np.random.default_rng(42)
    
    # Process each time window
    while actual_time < end_time+1 :
        window_end = actual_time + time_window
        
        # Get flows in this window
        window_mask = (eval_data_copy["end_time"] >= actual_time) & (eval_data_copy["end_time"] < window_end)
        window_flows = eval_data_copy[window_mask]
        
        if len(window_flows) == 0:
            # If no flows in window, jump to next window with data
            window_mask=(eval_data_copy["end_time"] >=window_end)
            window_flows = eval_data_copy[window_mask]
            actual_time = window_flows["end_time"].min()
            continue
        
        # Group by traffic class
        for traffic_class, class_group in window_flows.groupby("traffic_class"):
            class_indices = class_group.index
            class_p_cal = class_group["p_cal"].values
            
            # Compute aggregate probability
            agg_probability=np.mean(class_p_cal)
            
            prev_prob = previous_probs.get(traffic_class, 0.0)
            smoothed_prob = lambda_param * agg_probability + (1 - lambda_param) * prev_prob
            
            previous_probs[traffic_class] = smoothed_prob
            
            # All flows in this class get the same action
            actions[class_indices] = int(rng.random() < smoothed_prob)
        
        actual_time = window_end
    return actions


def bayesian_warden(p_cal,cost):
    """
    Cost-optimal action selection using Bayesian decision theory.
    
    For each flow, selects the action that minimizes expected cost,
    using thresholds computed from the cost matrix.
    
    Parameters:
        p_cal: array of calibrated covert probabilities
        cost: cost matrix
    
    Returns:
        array of optimal actions
    """
    actions = np.zeros(len(p_cal), dtype=int)
    
    # Compute thresholds between consecutive actions
    thresholds=[compute_threshold(i, i+1,cost) for i in range(len(cost)-1)]
    
    # For each flow, find the action with minimum expected cost
    for i in range(len(p_cal)):
        for j in range(len(thresholds)): 
            if p_cal[i] > thresholds[j]:
                actions[i] = j+1
            else:
                break
    return actions

def bayesian_warden_per_class(eval_data, p_cal, cost, time_window=1000.0, aggregation_method="mean", quantile=0.9,lambda_param=0.5):
    """
    Bayesian warden that applies actions per traffic class within time windows.
    
    Actions are determined by:
    1. Grouping flows by traffic class within each time window
    2. Computing aggregate probability (mean or quantile) per class
    3. Applying action decision based on this aggregate probability
    4. All flows in a class receive the same action
    
    Parameters:
        eval_data: pandas DataFrame with "flow_id", "end_time"
        p_cal: array of calibrated covert probabilities
        cost: cost matrix [action][true_label]
        time_window: duration of time window (default: 1000)
        aggregation_method: "mean" or "quantile" (default: "mean")
        quantile: quantile value if aggregation_method="quantile" (default: 0.9)
        lambda_param: smoothing factor (0, 1] (default: 0.5)
    
    Returns:
        array of optimal actions per flow
    """
    
    thresholds=[compute_threshold(i, i+1,cost) for i in range(len(cost)-1)]
    
    # Initialize results
    actions = np.zeros(len(eval_data), dtype=int)
    
    # Extract traffic classes
    eval_data_copy = eval_data.copy()
    eval_data_copy["traffic_class"] = eval_data_copy["flow_id"].apply(eval_traffic_class)
    eval_data_copy["p_cal"] = p_cal
    
    # Dictionary to store previous smoothed probability for each traffic class
    # p_{t-1}(E) for each class E
    previous_probs = {} 
     
    # Get time boundaries
    begin_time = eval_data_copy["end_time"].min()
    end_time = eval_data_copy["end_time"].max()
    
    actual_time = begin_time
    
    # Process each time window
    while actual_time < end_time+1 :
        window_end = actual_time + time_window
        
        # Get flows in this window
        window_mask = (eval_data_copy["end_time"] >= actual_time) & (eval_data_copy["end_time"] < window_end)
        window_flows = eval_data_copy[window_mask]
        
        if len(window_flows) == 0:
            # If no flows in window, jump to next window with data
            window_mask=(eval_data_copy["end_time"] >=window_end)
            window_flows = eval_data_copy[window_mask]
            actual_time = window_flows["end_time"].min()
            continue
        
        # Group by traffic class
        for traffic_class, class_group in window_flows.groupby("traffic_class"):
            class_indices = class_group.index
            class_p_cal = class_group["p_cal"].values
            
            # Compute aggregate probability
            if aggregation_method == "mean":
                agg_probability = np.mean(class_p_cal)
            else:  # quantile
                agg_probability = np.quantile(class_p_cal, quantile)
            
            prev_prob = previous_probs.get(traffic_class, 0.0)
            smoothed_prob = lambda_param * agg_probability + (1 - lambda_param) * prev_prob
            
            previous_probs[traffic_class] = smoothed_prob
            
            # Decide action based on aggregate probability
            action = 0  # default: allow
            for j in range(len(thresholds)):
                if smoothed_prob > thresholds[j]:
                    action = j + 1
                else:
                    break
            
            # All flows in this class get the same action
            actions[class_indices] = action
        
        actual_time = window_end
    
    return actions

def update_cost(cost,alpha_legit,alpha_covert,feedback_signals):
    """
    Update running cost estimates using exponential smoothing.
    
    Also enforces ordering constraints.
    
    Parameters:
        cost: cost matrix [action][true_label] to update
        alpha_legit: learning rate for legitimate cost (0, 1]
        alpha_covert: learning rate for covert cost (0, 1]
        feedback_signals: dict mapping action -> {"c_legit": [...], "c_covert": [...]}
    
    Returns:
        updated cost matrix
    """
    for action, signals in feedback_signals.items(): 
        c_legit = np.mean(signals["c_legit"]) if signals["c_legit"] else 0
        c_covert = np.mean(signals["c_covert"]) if signals["c_covert"] else 0

        # Update using exponential smoothing
        cost[action][0] = max(0,min(1,((1 - alpha_legit) * cost[action][0] + alpha_legit * c_legit)))
        cost[action][1] = max(0,min(1,((1 - alpha_covert) * cost[action][1] + alpha_covert * c_covert)))
        
        # Enforce ordering: legitimate cost should decrease with action severity
        for action in range(len(cost) - 1):
            if cost[action][0] >= cost[action + 1][0]:
                cost[action][0], cost[action + 1][0] = cost[action + 1][0], cost[action][0]
        
        # Enforce ordering: covert cost should increase with action severity
        for action in range(len(cost) - 1):
            minus = len(cost) - action - 1
            if cost[minus][1] >= cost[minus - 1][1]:
                cost[minus][1], cost[minus - 1][1] = cost[minus - 1][1], cost[minus][1]

    return cost

def adaptive_bayesian_warden_per_class(eval_data, p_cal, base_cost, alpha_legit, alpha_covert, time_window=1000.0, aggregation_method="mean", quantile=0.9,lambda_param=0.5):
    """
    Run adaptive Bayesian warden on evaluation data with time windows.
    Maintains running estimates of costs that adapt based on observed feedback.
    Thresholds are recomputed after each window based on updated costs.
    
    Parameters:
        eval_data: pandas DataFrame with "flow_id", "end_time", "label"
        p_cal: array of calibrated covert probabilities
        base_cost: initial cost matrix [action][true_label]
        alpha_legit: learning rate for legitimate cost (0, 1]
        alpha_covert: learning rate for covert cost (0, 1]
        time_window: duration of time window in seconds (default: 10000s)
        aggregation_method: "mean" or "quantile" (default: "mean")
        quantile: quantile value if aggregation_method="quantile" (default: 0.9)
        lambda_param: smoothing factor (0, 1] (default: 0.5)
    
    Returns:
        tuple: (actions array, cost evolution list, thresholds evolution list)
    """
    n_thresholds = len(base_cost) - 1
    thresholds = [compute_threshold(i, i + 1, base_cost) for i in range(n_thresholds)]
    
    # Prepare data
    eval_data_copy = eval_data.copy()
    eval_data_copy["traffic_class"] = eval_data_copy["flow_id"].apply(eval_traffic_class)
    eval_data_copy["p_cal"] = p_cal
    
    # Create deep copy of cost to avoid modifying original
    cost={k: v.copy() for k, v in base_cost.items()}
    
    # Dictionary to store previous smoothed probability for each traffic class
    # p_{t-1}(E) for each class E
    previous_probs = {}
    
    # Get time boundaries
    begin_time = eval_data_copy["end_time"].min()
    end_time = eval_data_copy["end_time"].max()
    
    # Initialize result structures
    actions = np.zeros(len(eval_data), dtype=int)
    save_cost = []
    save_thresholds = []
    
    actual_time = begin_time
    # Process each time window
    while actual_time < end_time +1:
        feedback_signals = {a: {"c_legit": [], "c_covert": []} for a in range(len(cost))}
        window_end = actual_time + time_window
        
        # Get flows in this window
        window_mask = (eval_data_copy["end_time"] >= actual_time) & (eval_data_copy["end_time"] < window_end)
        window_flows = eval_data_copy[window_mask]
        
        if len(window_flows) == 0:
            # If no flows in window, jump to next window with data
            window_mask=(eval_data_copy["end_time"] >=window_end)
            window_flows = eval_data_copy[window_mask]
            actual_time = window_flows["end_time"].min()
            continue

        # Extract probabilities and labels for this window
        p_cal_window=window_flows["p_cal"]
        y_true_window=window_flows["label"]
        
        # Group by traffic class
        for traffic_class, class_group in window_flows.groupby("traffic_class"):
            class_indices = class_group.index
            class_p_cal = p_cal_window[class_indices]
            class_y_true = y_true_window[class_indices]

            # Compute π_t(E) = fraction of covert flows in traffic class E
            pi_t = class_y_true.mean()
            
            # Compute aggregate probability
            if aggregation_method == "mean":
                agg_probability = np.mean(class_p_cal)
            else:  # quantile
                agg_probability = np.quantile(class_p_cal, quantile)
            
            prev_prob = previous_probs.get(traffic_class, 0.0)
            smoothed_prob = lambda_param * agg_probability + (1 - lambda_param) * prev_prob
            
            previous_probs[traffic_class] = smoothed_prob
            
            # Decide action based on aggregate probability and current thresholds
            action = 0  # default: allow
            for j in range(n_thresholds):
                if smoothed_prob > thresholds[j]:
                    action = j + 1
                else:
                    break

            # Compute class-conditional observations for feedback
            c_legit_obs = (1 - pi_t) * cost[action][0]
            c_covert_obs = pi_t * cost[action][1]

            # Accumulate feedback for this action
            feedback_signals[action]["c_legit"].append(c_legit_obs)
            feedback_signals[action]["c_covert"].append(c_covert_obs)

            # All flows in this class get the same action
            actions[class_indices] = action
        #Store current costs before update
        save_cost.append({k: v.copy() for k, v in cost.items()})
        save_thresholds.append(thresholds.copy())
        
        # Only update if enough feedback samples (> 20)
        if(len(window_flows)>20):    
            cost=update_cost(cost,alpha_legit,alpha_covert,feedback_signals)
            thresholds=[compute_threshold(i, i+1,cost) for i in range(n_thresholds)]
        actual_time=window_end
    save_cost.append({k: v.copy() for k, v in cost.items()})
    save_thresholds.append(thresholds.copy())

    return actions,save_cost,save_thresholds

def continuous_cost_evaluation(actions,cost,y_true):
    """
    Evaluate cumulative cost of actions against true labels.
    
    Computes total cost incrementally, returning list of cumulative costs.
    
    Parameters:
        actions: array of chosen actions
        cost: cost matrix
        y_true: array of true labels (0=legitimate, 1=covert)
    
    Returns:
        list: cumulative cost at each flow index
    """
    total_cost=0
    total_cost_list=[]

    # Sum costs incrementally for each flow
    for i in range(len(actions)):
        if y_true[i]==0:
            total_cost+=(cost[actions[i]][0])
        else :
            total_cost+=(cost[actions[i]][1])
        total_cost_list.append(total_cost)
    
    return total_cost_list


def calibrate_model(train_data):
    """
    Train and calibrate the warden model on training data.
    
    Steps:
    1. Split data into scoring (70%) and calibration (30%) sets
    2. Extract feature statistics (min, Q1, median, Q3, max)
    4. Compute anomaly scores on calibration set
    5. Train calibrator to convert scores to probabilities
    
    Parameters:
        train_data: pandas DataFrame with features and "label" column
    
    Returns:
        dict: model containing weights, feature statistics, and calibrator
    """
    
    score_data, calib_data = train_test_split(train_data,test_size=0.3,stratify=train_data["label"],random_state=42)
    
    # Better score
    score_y_true=score_data["label"]
    calib_y_true=calib_data["label"]
    
    
    # Compute statistics for each feature
    stats={
        feature:{
            "m":score_data[feature].min(),
            "Q1":score_data[feature].quantile(0.25),
            "Median":score_data[feature].quantile(0.5),
            "Q3":score_data[feature].quantile(0.75),
            "M":score_data[feature].max()
        }
        for feature in feature_cols
    }
    
    # Compute anomaly scores for all calibration samples
    scores = calib_data["suspicion_score"].values
    
    # Train calibrator: logistic regression mapping scores to probabilities
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(scores.reshape(-1,1), calib_y_true)

    return {
        "features_stats": stats,
        "calibrator": calibrator
    }


def plot_probability_distribution(p_cal,y_true,cost_2_action,cost_3_action):
    """
    Visualize the distribution of calibrated covert probabilities.
    
    Shows separate distributions for legitimate and covert flows, along with decision thresholds for different cost matrices.
    
    Parameters:
        p_cal: array of calibrated covert probabilities
        y_true: array of true labels
        cost_2_action: 2-action cost matrix
        cost_3_action: 3-action cost matrix
    """
    x=np.arange(len(y_true))

    threshold_2_actions=compute_threshold(0, 1,cost_2_action)
    thresholds_3_actions=[compute_threshold(i, i+1,cost_3_action) for i in range(2)]
    
    plt.figure(figsize=(7,6))
    
    # Plot probability distributions
    plt.scatter(x[y_true==1], p_cal[y_true==1],s=10,c="r",label="Covert flows")
    plt.scatter(x[y_true==0], p_cal[y_true==0],s=10,c="b",label="Legitimate flows")
    # Plot decision thresholds
    plt.plot(x,[threshold_2_actions]*len(x),"k--",label="2 action threshold")
    plt.plot(x,[thresholds_3_actions[0]]*len(x),color="gray",linestyle="--",label="Capture threshold")
    plt.plot(x,[thresholds_3_actions[1]]*len(x),color="brown",linestyle="--",label="Normalize threshold")
    
    plt.xlabel("Flow index")
    plt.ylabel("Estimated covert probability")
    plt.title("Distribution of calibrated covert probabilities")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost_comparison(total_cost_evolution,y_true,cost_2_action):
    """
    Compare cumulative cost over flows for different warden strategies.
    
    Plots evolution of total cost as flows are processed in order,
    showing performance differences between naive and Bayesian approaches.
    """
    # Compute theoretical minimum: apply action matching true label
    minimum_bound = continuous_cost_evaluation(y_true, cost_2_action, y_true)
    
    x=np.arange(len(y_true))
    
    plt.figure(figsize=(7,6))
    
    # Plot cumulative costs for each warden
    for name,total_cost_list in total_cost_evolution.items():
        plt.plot(x, total_cost_list, label=name)
    # Plot theoretical minimum
    plt.plot(x, minimum_bound, "k--", label="Minimum cumulative cost obtainable")
    
    plt.xlabel("Flow index")
    plt.ylabel("Cumulative cost")
    #plt.title("Cumulative cost comparison: warden strategies")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost_evolution(cost_evolution,base_cost,name,action_name):
    """
    Plot evolution of legitimate and covert costs across time windows.
    
    Shows how running cost estimates adapt over time, compared to base costs.
    
    Parameters:
        cost_evolution: list of cost matrices (one per window)
        base_cost: initial cost matrix (for reference)
        name: name of the warden for plot title
        action_name: list of action names ["ALLOW", "CAPTURE", "NORMALIZE"]
    """
    nb_action=len(cost_evolution[0])
    nb_windows=len(cost_evolution)

    x=np.arange(nb_windows)
    ys_legit=[[cost[action][0] for cost in cost_evolution] for action in range(nb_action)]
    ys_covert=[[cost[action][1] for cost in cost_evolution] for action in range(nb_action)]

    plt.figure(figsize=(8,4))
    
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })  

    # Plot cost evolution for each action
    for i in range(nb_action):
        plt.plot(x, ys_legit[i], label=f"Cost for {action_name[i]} on legit traffic")
        plt.plot(x, ys_covert[i], label=f"Cost for {action_name[i]} on covert traffic")
        
    plt.xlabel("Windows")
    plt.ylabel("Cost value")
    plt.title(f"Cost evolution for {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_thresholds_evolution(thresholds_evolution,base_thresholds,name,action_name):
    """
    Plot evolution of decision thresholds across time windows.
    
    Shows how decision boundaries adapt based on updated costs.
    
    Parameters:
        thresholds_evolution: list of threshold arrays (one per window)
        base_thresholds: initial thresholds (for reference)
        name: name of the warden for plot title
        action_name: list of action names ["ALLOW", "CAPTURE", "NORMALIZE"]
    """
    nb_thresholds=len(thresholds_evolution[0])
    nb_windows=len(thresholds_evolution)

    x=np.arange(nb_windows)
    ys=[[thresholds[i] for thresholds in thresholds_evolution] for i in range(nb_thresholds)]

    plt.figure(figsize=(8,4))
    
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })  

    # Plot threshold evolution for each threshold
    for i in range(nb_thresholds):
        plt.plot(x, ys[i], label=f"Threshold {action_name[i]}-{action_name[i+1]}")
    
    plt.xlabel("Windows")
    plt.ylabel("Threshold value")
    plt.title(f"Thresholds evolution for {name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_distribution(name, actions, y_true):
    """
    Helper function to print action distribution.
    
    Prints how many flows of each type (legitimate/covert) receive each action.
    
    Parameters:
        name: name of the warden
        actions: array of actions taken
        y_true: array of true labels
    """
    legit_actions = actions[y_true == 0]
    covert_actions = actions[y_true == 1]
    
    unique_legit, counts_legit = np.unique(legit_actions, return_counts=True)
    unique_covert, counts_covert = np.unique(covert_actions, return_counts=True)
    
    nb_legit=sum(counts_legit)
    nb_covert=sum(counts_covert)
    
    print(f"{name} (legitimate flows):")
    for action, count in zip(unique_legit, counts_legit):
        print(f"  Action {action}: {count}")
    
    print(f"{name} (covert flows):")
    for action, count in zip(unique_covert, counts_covert):
        print(f"  Action {action}: {count}")
    
    print("FPR:",counts_legit[-1]/nb_legit)
    print("TPR:",counts_covert[-1]/nb_covert)
    print()

def main():
    """
    Main function: orchestrates model training and warden comparison.
    
    Workflow:
    1. Load and calibrate model on training data
    2. Evaluate on test data using different warden strategies:
       - Naive warden (random)
       - Bayesian warden (individual flows)
       - Per-class Bayesian (mean aggregation)
       - Per-class Bayesian (quantile aggregation)
       - Adaptive per-class Bayesian (mean aggregation)
       - Adaptive per-class Bayesian (quantile aggregation)
    3. Compare costs and visualize results
    4. Plot cost and threshold evolution for adaptive wardens
    """

    parser = argparse.ArgumentParser(
        description="Warden calibration and evaluation tool"
    )

    parser.add_argument("-td", "--train_dataset",
                        help="Training dataset path")

    parser.add_argument("-ed", "--eval_dataset",
                        help="Evaluation dataset path")

    parser.add_argument("-c", "--cost",
                        help="Cost dictionary file path")

    args = parser.parse_args()

    # Default paths
    DEFAULT_TRAIN = "datasets\\cic_dns_exf\\storage\\medium\\train.csv"
    DEFAULT_EVAL = "datasets\\cic_dns_exf\\storage\\medium\\test.csv"

    # Load datasets
    train_path = args.train_dataset if args.train_dataset else DEFAULT_TRAIN
    eval_path = args.eval_dataset if args.eval_dataset else DEFAULT_EVAL
    
    print("--------- DATASET ---------\n")
    print(f"[*] Training dataset: {train_path}")
    print(f"[*] Evaluation dataset: {eval_path}")

    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)

    # Train and calibrate model
    model = calibrate_model(train_data)
    
    # omega=model["weight"]
    # stats=model["features_stats"]
    calibrator=model["calibrator"]

    eval_y_true=eval_data["label"].values
    
    # Compute anomaly scores and convert to probabilities
    #scores = np.array([score_evaluation(eval_data.iloc[i],stats,omega) for i in range(len(eval_data))])
    scores = eval_data["suspicion_score"].values

    fpr, tpr, thresholds = roc_curve(eval_y_true, scores)
     
    plt.figure(figsize=(7,6 ))
    
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })  
   
    plt.plot(fpr, tpr )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    p_cal = calibrator.predict_proba(scores.reshape(-1,1))[:,1]
    
    # Visualize probability distribution
    plot_probability_distribution(p_cal,eval_y_true,COST_2_ACTION,COST_3_ACTION)

    # Generate decisions from different wardens
    naive_warden_actions=per_class_naive_warden(eval_data,p_cal)
    #bayesian_warden_actions_2=bayesian_warden(p_cal,COST_2_ACTION)
    #bayesian_warden_actions_3=bayesian_warden(p_cal,COST_3_ACTION)
    bayesian_warden_actions_2_mean=bayesian_warden_per_class(eval_data,p_cal,COST_2_ACTION)
    bayesian_warden_actions_3_mean=bayesian_warden_per_class(eval_data,p_cal,COST_3_ACTION)
    bayesian_warden_actions_2_quantile=bayesian_warden_per_class(eval_data,p_cal,COST_2_ACTION,aggregation_method="quantile")
    bayesian_warden_actions_3_quantile=bayesian_warden_per_class(eval_data,p_cal,COST_3_ACTION,aggregation_method="quantile")
    adaptive_bayesian_warden_actions_2_mean,cost_adaptive_bayesian_warden_actions_2_mean,thresholds_adaptive_bayesian_warden_actions_2_mean=adaptive_bayesian_warden_per_class(eval_data,p_cal,COST_2_ACTION,0.05,0.05)
    adaptive_bayesian_warden_actions_3_mean,cost_adaptive_bayesian_warden_actions_3_mean,thresholds_adaptive_bayesian_warden_actions_3_mean=adaptive_bayesian_warden_per_class(eval_data,p_cal,COST_3_ACTION,0.05,0.05)
    adaptive_bayesian_warden_actions_2_quantile,cost_adaptive_bayesian_warden_actions_2_quantile,thresholds_adaptive_bayesian_warden_actions_2_quantile=adaptive_bayesian_warden_per_class(eval_data,p_cal,COST_2_ACTION,0.05,0.05,aggregation_method="quantile")
    adaptive_bayesian_warden_actions_3_quantile,cost_adaptive_bayesian_warden_actions_3_quantile,thresholds_adaptive_bayesian_warden_actions_3_quantile=adaptive_bayesian_warden_per_class(eval_data,p_cal,COST_3_ACTION,0.05,0.05,aggregation_method="quantile")


    # Organize warden results
    warden_name={
        "mean Bayesian":[bayesian_warden_actions_2_mean,bayesian_warden_actions_3_mean],
        "quantile Bayesian":[bayesian_warden_actions_2_quantile,bayesian_warden_actions_3_quantile],
        "adaptive mean Bayesian":[adaptive_bayesian_warden_actions_2_mean,adaptive_bayesian_warden_actions_3_mean],
        "adaptive quantile Bayesian":[adaptive_bayesian_warden_actions_2_quantile, adaptive_bayesian_warden_actions_3_quantile]
    }
    
    cost_evolution={
        "adaptive mean Bayesian":[cost_adaptive_bayesian_warden_actions_2_mean,cost_adaptive_bayesian_warden_actions_3_mean],
        "adaptive quantile Bayesian":[cost_adaptive_bayesian_warden_actions_2_quantile,cost_adaptive_bayesian_warden_actions_3_quantile]
    }

    thresholds_evolution={
        "adaptive mean Bayesian":[thresholds_adaptive_bayesian_warden_actions_2_mean,thresholds_adaptive_bayesian_warden_actions_3_mean],
        "adaptive quantile Bayesian":[thresholds_adaptive_bayesian_warden_actions_2_quantile,thresholds_adaptive_bayesian_warden_actions_3_quantile]
    }

    # Analyze action distribution
    print("\n\n--------- ACTION DISTRIBUTION ---------\n")
    print_distribution("naive warden", naive_warden_actions, eval_y_true)
    for name,action_list in warden_name.items():
        print_distribution(name+" (2 actions)", action_list[0], eval_y_true)
        print_distribution(name+" (3 actions)", action_list[1], eval_y_true)

    # Evaluate total cost
    total_cost_evolution={"naive": continuous_cost_evaluation(naive_warden_actions,COST_2_ACTION,eval_y_true)}
    for name,action_list in warden_name.items():
        total_cost_evolution[name +" with 2 actions"]=continuous_cost_evaluation(action_list[0],COST_2_ACTION,eval_y_true)
        total_cost_evolution[name +" with 3 actions"]=continuous_cost_evaluation(action_list[1],COST_3_ACTION,eval_y_true)
        
    print("--------- AVERAGE COST ---------\n")
    for name,total_cost_list in total_cost_evolution.items():
        print(name+":",total_cost_list[-1]/327821)
    
    # Visualize comparisons
    plot_cost_comparison(total_cost_evolution,eval_y_true,COST_2_ACTION)

    # Plot cost evolution for adaptive wardens
    for name,cost_list in cost_evolution.items():
        plot_cost_evolution(cost_list[0],COST_2_ACTION,name+" (2 actions)",["ALLOW", "NORMALIZE"])
        plot_cost_evolution(cost_list[1],COST_3_ACTION,name+" (3 actions)",["ALLOW", "CAPTURE", "NORMALIZE"])

    # Plot threshold evolution for adaptive wardens
    for name,thresholds_list in thresholds_evolution.items():
        plot_thresholds_evolution(thresholds_list[0],thresholds_list[0],name+" (2 actions)",["ALLOW", "NORMALIZE"])
        plot_thresholds_evolution(thresholds_list[1],thresholds_list[1][0],name+" (3 actions)",["ALLOW", "CAPTURE", "NORMALIZE"])
    
    return 0


if __name__ == "__main__":
    main()

