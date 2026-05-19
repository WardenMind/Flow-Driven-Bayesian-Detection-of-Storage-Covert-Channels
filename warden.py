import argparse
import ast
import importlib
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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

def feature_normalisation(col,stat_feature):
    """
    Normalize a feature column to [0, 1] based on interquartile range or min-max.
    
    Parameters:
        col: numpy array of feature values
        stat_feature: dict with "m" (min), "M" (max), "Q1", "Q3", "Median"
    
    Returns:
        numpy array: normalized values in [0, 1]
    """
    n=len(col)
    norm=[0]*n
    if stat_feature["Q3"]!=stat_feature["Q1"]:
        for i in range(n):
            norm[i]=abs((col[i]-stat_feature["Median"])/(stat_feature["Q3"]-stat_feature["Q1"])) 
    else:
        for i in range(n):
            norm[i]=abs((col[i]-stat_feature["m"])/(stat_feature["M"]-stat_feature["m"])) if stat_feature["M"]!=stat_feature["m"] else col[i]-stat_feature["m"]
    
    # Clamp to [0, 1]
    return np.array([max(0,min(1,norm[i])) for i in range(n)])

def score_evaluation(row,stats,omega):
    """
    Compute anomaly score for a single flow.
    
    Uses normalized deviations from the median for each feature,
    weighted by feature importance (omega).
    
    Parameters:
        row: pandas Series containing feature values for one flow
        stats: dict with statistical info (Q1, Q3, min, max, median) per feature
        omega: dict of feature weights from logistic regression model
    
    Returns:
        float: aggregated anomaly score between 0 and 1
    """
    score=0
    alpha={}

    # Normalize each feature to [0, 1] based on interquartile range or min-max
    for feature in feature_cols:
        stat_feature=stats[feature]

        # Use interquartile range (IQR) if Q1 != Q3, otherwise use min-max
        if stat_feature["Q3"]!=stat_feature["Q1"]:
            alpha[feature]=abs((row[feature]-stat_feature["Median"])/(stat_feature["Q3"]-stat_feature["Q1"])) 
        else:
            alpha[feature]=abs((row[feature]-stat_feature["m"])/(stat_feature["M"]-stat_feature["m"])) if stat_feature["M"]!=stat_feature["m"] else row[feature]-stat_feature["m"]
        # Clamp to [0, 1]
        alpha[feature]=max(0,min(1,alpha[feature]))
    # Compute weighted sum of normalized deviations
    for feature in feature_cols:
        score+=(omega[feature]*alpha[feature])
    return score

def compute_threshold(a, b, cost):
    """
    Compute decision threshold between two actions using cost analysis.
    
    The threshold P* is where the expected cost is equal:

    Parameters:
        a, b: action indices
        cost: cost matrix [action][true_label]
    
    Returns:
        float or None: probability threshold for choosing action b over a
    """
    num = cost[a][0] - cost[b][0]
    den = (cost[b][1] - cost[b][0]) - (cost[a][1] - cost[a][0])
    return num / den if den != 0 else None

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


def bayesian_warden_per_class(eval_data, p_cal, thresholds, time_window=100000.0, aggregation_method="mean", quantile=0.9,lambda_param=0.5):
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
        thresholds: list of decision thresholds between consecutive actions
        time_window: duration of time window (default: 1000)
        aggregation_method: "mean" or "quantile" (default: "mean")
        quantile: quantile value if aggregation_method="quantile" (default: 0.9)
        lambda_param: smoothing factor (0, 1] (default: 0.5)
    
    Returns:
        array of optimal actions per flow
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
        # Only update if enough feedback samples (> 5)
        if len(signals["c_legit"]) + len(signals["c_covert"])>5: 
            c_legit = np.mean(signals["c_legit"]) if signals["c_legit"] else 0
            c_covert = np.mean(signals["c_covert"]) if signals["c_covert"] else 0
    
            # Update using exponential smoothing
            cost[action][0] = max(0,min(1,((1 - alpha_legit) * cost[action][0] + alpha_legit * c_legit)))
            cost[action][1] = max(0,min(1,((1 - alpha_covert) * cost[action][1] + alpha_covert * c_covert)))
            
            # Enforce ordering: legitimate costs should decrease with action severity
            for action in range(len(cost) - 1):
                if cost[action][0] >= cost[action + 1][0]:
                    cost[action][0], cost[action + 1][0] = cost[action + 1][0], cost[action][0]
            
            # Enforce ordering: covert costs should increase with action severity
            for action in range(len(cost) - 1):
                minus = len(cost) - action - 1
                if cost[minus][1] >= cost[minus - 1][1]:
                    cost[minus][1], cost[minus - 1][1] = cost[minus - 1][1], cost[minus][1]

    return cost


def adaptive_bayesian_warden_per_class(eval_data, p_cal, base_cost, thresholds, alpha_legit, alpha_covert, time_window=100000.0, aggregation_method="mean", quantile=0.9,lambda_param=0.5):
    """
    Run adaptive Bayesian warden on evaluation data with time windows.
    
    Maintains running estimates of costs that adapt based on observed feedback.
    Thresholds are recomputed after each window based on updated costs.
    
    Parameters:
        eval_data: pandas DataFrame with "flow_id", "end_time", "label"
        p_cal: array of calibrated covert probabilities
        base_cost: initial cost matrix [action][true_label]
        thresholds: initial decision thresholds
        alpha_legit: learning rate for legitimate cost (0, 1]
        alpha_covert: learning rate for covert cost (0, 1]
        time_window: duration of time window in seconds (default: 10000s)
        aggregation_method: "mean" or "quantile" (default: "mean")
        quantile: quantile value if aggregation_method="quantile" (default: 0.9)
        lambda_param: smoothing factor (0, 1] (default: 0.5)
    
    Returns:
        tuple: (actions array, updated cost matrix, updated thresholds)
    """
    
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

    copy_thresholds=thresholds.copy()
    n_thresholds=len(copy_thresholds)
    
    # Initialize result structures
    actions = np.zeros(len(eval_data), dtype=int)
    
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
                if smoothed_prob > copy_thresholds[j]:
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
        cost=update_cost(cost,alpha_legit,alpha_covert,feedback_signals)
        copy_thresholds=[compute_threshold(i, i+1,cost) for i in range(n_thresholds)]
        actual_time=window_end
    
    return actions, cost,copy_thresholds


def cost_evaluation(actions,y_true,cost):
    """
    Evaluate total cost of actions against true labels.
    
    Parameters:
        actions: array of chosen actions
        y_true: array of true labels (0=legitimate, 1=covert)
        cost: cost matrix
    
    Returns:
        float: total cost incurred
    """
    legit_actions=actions[y_true==0]
    covert_actions=actions[y_true==1]
    total_cost=0
    
    # Sum costs for legitimate flows
    for action in legit_actions:
        total_cost+=cost[action][0]
    
    # Sum costs for covert flows
    for action in covert_actions:
        total_cost+=cost[action][1]
    
    return total_cost

def load_cost(path):
    """
    Load custom cost matrix and adaptation rates from a Python file.
    
    Expected file format: contains COST, alpha_legit, and alpha_covert variables.
    Parameters:
        path: path to Python file with cost matrix and parameters
    
    Returns:
        tuple: (cost_matrix, alpha_legit, alpha_covert)
    """
    spec = importlib.util.spec_from_file_location("cost_module", path)
    cost_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cost_module)
    return cost_module.COST, cost_module.alpha_legit,cost_module.alpha_covert


def load_warden(path):
    """
    Load a trained warden model from disk using joblib.
    
    Parameters:
        path: path to saved .joblib model file
    
    Returns:
        dict: model containing weights, stats, calibrator, thresholds, costs, and alphas
    """
    with open(path, "rb") as f:
        model = load(f)
    return model


def save_warden(path, model):
    """
    Save a trained warden model to disk using joblib.
    
    Parameters:
        path: path where to save the model
        model: model dict to save
    """
    with open(path, "wb") as f:
        dump(model, f)


def calibrate_model(train_data, cost, alpha_legit, alpha_covert):
    """
    Train and calibrate the warden model on training data.
    
    Steps:
    1. Split data into scoring (70%) and calibration (30%) sets
    2. Extract feature statistics (min, Q1, median, Q3, max)
    3. Normalize features and train logistic regression for weights
    4. Compute anomaly scores on calibration set
    5. Train calibrator to convert scores to probabilities
    
    Parameters:
        train_data: pandas DataFrame with features and "label" column
        cost: cost matrix
        alpha_legit: learning rate for legitimate cost adaptation
        alpha_covert: learning rate for covert cost adaptation
    
    Returns:
        dict: model containing weights, feature statistics, calibrator, decision thresholds, cost matrix, and adaptation rates
    """
    score_data, calib_data = train_test_split(train_data,test_size=0.3,stratify=train_data["label"],random_state=42)
    
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
    
    # Normalize features and train logistic regression for importance weights
    X_norm=pd.DataFrame({feature:feature_normalisation(np.array(score_data[feature]),stats[feature]) for feature in feature_cols})

    model = LogisticRegression()
    model.fit(X_norm, score_y_true)

    # Extract absolute coefficient values as feature weights
    omega={
        feature_cols[i]: abs(model.coef_[0][i]) for i in range(len(feature_cols))
    }

    total_weight = sum(omega.values())
    for feature in omega:
        omega[feature] /= total_weight
        
    # Compute anomaly scores for all calibration samples
    scores=np.array([score_evaluation(calib_data.iloc[i],stats,omega) for i in range(len(calib_data))])
    
    # Train calibrator: logistic regression mapping scores to probabilities
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(scores.reshape(-1,1), calib_y_true)

    # Compute decision thresholds between consecutive actions
    thresholds = [compute_threshold(i, i + 1, cost) for i in range(len(cost) - 1)]
    
    return {
        "weight": omega,
        "features_stats": stats,
        "calibrator": calibrator,
        "thresholds": thresholds,
        "cost": cost,
        "alpha_legit": alpha_legit,
        "alpha_covert": alpha_covert
    }
    

def evaluate(model, eval_data, quantile, adaptation):
    """
    Evaluate the warden model on evaluation data.
    
    Supports two aggregation methods (mean or quantile) and optional adaptation.
    
    Parameters:
        model: calibrated warden model (from calibrate_model)
        eval_data: pandas DataFrame with features, 'label', 'flow_id', 'end_time'
        quantile: boolean, if True uses 90th percentile instead of mean
        adaptation: boolean, if True adapts costs over time windows
    
    Returns:
        tuple: (actions array, updated cost matrix, updated thresholds)
    """
    omega=model["weight"]
    stats=model["features_stats"]
    calibrator=model["calibrator"]

    # Compute anomaly scores
    scores = np.array([score_evaluation(eval_data.iloc[i],stats,omega) for i in range(len(eval_data))])

    # Convert scores to calibrated probabilities
    p_cal = calibrator.predict_proba(scores.reshape(-1,1))[:,1]

    if quantile:
        if adaptation:
            # Adaptive warden with quantile aggregation
            action, updated_cost, updated_thresholds = adaptive_bayesian_warden_per_class(eval_data,p_cal,model["cost"],model["thresholds"],model["alpha_legit"],model["alpha_covert"],aggregation_method="quantile")
        else:
            # Non-adaptive warden with quantile aggregation
            action = bayesian_warden_per_class(eval_data,p_cal,model["thresholds"],aggregation_method="quantile")
            updated_cost = model["cost"]
            updated_thresholds = model["thresholds"]
    else:
        if adaptation:
            # Adaptive warden with mean aggregation
            action, updated_cost, updated_thresholds = adaptive_bayesian_warden_per_class(eval_data,p_cal,model["cost"],model["thresholds"],model["alpha_legit"],model["alpha_covert"])
        else:
            # Non-adaptive warden with mean aggregation
            action = bayesian_warden_per_class(eval_data,p_cal,model["thresholds"])
            updated_cost = model["cost"]
            updated_thresholds = model["thresholds"]

    return action, updated_cost, updated_thresholds


def main():
    """
    Main function: orchestrates warden training and evaluation.
    
    Supports two modes:
    1. TRAINING MODE: Train a new model and evaluate it
    2. WARDEN MODE: Load pre-trained model and evaluate it
    
    Supports optional flags:
    - quantile: use 90th percentile instead of mean for aggregation
    - adaptation: adapt costs over time windows during evaluation
    """

    parser = argparse.ArgumentParser(
        description="Adaptive Bayesian Warden calibration and evaluation tool"
    )

    parser.add_argument("-td", "--train_dataset",
                        help="Training dataset path")

    parser.add_argument("-ed", "--eval_dataset",
                        help="Evaluation dataset path")

    parser.add_argument("-c", "--cost",
                        help="Cost dictionary file path")

    parser.add_argument("-w", "--warden",
                        help="Pre-calibrated warden file path")
    
    parser.add_argument("-sw", "--save_warden",
                        help="Save calibrated in file path")
    
    parser.add_argument("-q", "--quantile", action="store_true",
                        help="Aggregate value probability with high quantile (90%)")
    
    parser.add_argument("-a", "--adaptation", action="store_true",
                        help="Adapt costs over time windows during evaluation")

    args = parser.parse_args()

    # Default paths
    DEFAULT_TRAIN = "datasets\\cic_dns_exf\\storage\\medium\\train.csv"
    DEFAULT_EVAL = "datasets\\cic_dns_exf\\storage\\medium\\test.csv"
    DEFAULT_COST = "cost_default.py"
    
    # ===== WARDEN MODE: Evaluate using pre-trained model =====
    if args.warden:
        print("[*] Loading pre-calibrated warden model...")
        model = load_warden(args.warden)

        eval_path = args.eval_dataset if args.eval_dataset else DEFAULT_EVAL
        eval_data = pd.read_csv(eval_path)
        actions,updated_cost, updated_thresholds=evaluate(model, eval_data, args.quantile, args.adaptation)
        
        print("\n[*] Results:")
        print(f"    Actions taken (first 10): {actions[:10]}")
        print(f"    Updated cost: {updated_cost}")
        print(f"    Updated thresholds: {updated_thresholds}")
        print(f"    Total cost: {cost_evaluation(actions,eval_data["label"].values, model["cost"])}")

        if args.save_warden:
            print(f"\n[*] Saving warden model to {args.save_warden}.joblib...")
            if args.adaptation:
                model["cost"]=updated_cost
                model["thresholds"]=updated_thresholds
            save_warden(args.save_warden,model)
            print("[*] Model saved successfully!")

        return 0

    # STANDARD MODE

    train_path = args.train_dataset if args.train_dataset else DEFAULT_TRAIN
    eval_path = args.eval_dataset if args.eval_dataset else DEFAULT_EVAL
    cost_path = args.cost if args.cost else DEFAULT_COST

    print(f"[*] Training dataset: {train_path}")
    print(f"[*] Evaluation dataset: {eval_path}")
    print(f"[*] Cost file: {cost_path}\n")

    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)
    cost, alpha_legit, alpha_covert= load_cost(cost_path)

    model = calibrate_model(train_data, cost, alpha_legit, alpha_covert)

    print(f"[*] Initial cost matrix: {cost}")
    print(f"[*] Initial thresholds: {model['thresholds']}\n")

    actions,updated_cost, updated_thresholds=evaluate(model, eval_data, args.quantile, args.adaptation)
    
    print("--------- EVALUATION ---------\n")
    print(f"    Actions taken (first 10): {actions[:10]}")
    print("[*] Results:")
    print(f"    Updated cost: {updated_cost}")
    print(f"    Updated thresholds: {updated_thresholds}")
    print(f"    Total cost: {cost_evaluation(actions,eval_data["label"].values, model["cost"])}")
    if args.save_warden:
        print(f"\n[*] Saving warden model to {args.save_warden}.joblib...")
        if args.adaptation:
            model["cost"]=updated_cost
            model["thresholds"]=updated_thresholds
        save_warden(args.save_warden,model)
        print("[*] Model saved successfully!")
    
    return 0


if __name__ == "__main__":
    main()