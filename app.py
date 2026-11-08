import os
import json
import uuid
import hashlib
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

BLOCKCHAIN_NETWORK = os.getenv("BLOCKCHAIN_NETWORK", "ganache").lower()

BLOCKCHAIN_RPC_URL = os.getenv(
    "BLOCKCHAIN_RPC_URL",
    os.getenv("MEDICHAIN_GANACHE_URL", "http://127.0.0.1:7545")
)

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet, InvalidToken

from web3 import Web3
from solcx import compile_source, install_solc, get_installed_solc_versions

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "medichain.db"
KEY_PATH = BASE_DIR / "medichain.key"
DATASET_PATH = BASE_DIR / "datasets" / "synthetic_medical_symptoms_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "disease_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# BLOCKCHAIN CONFIGURATION
# ============================================================

BLOCKCHAIN_NETWORK = os.getenv(
    "BLOCKCHAIN_NETWORK",
    "ganache"
).lower()

BLOCKCHAIN_RPC_URL = os.getenv(
    "BLOCKCHAIN_RPC_URL",
    os.getenv(
        "MEDICHAIN_GANACHE_URL",
        "http://127.0.0.1:7545"
    )
)

BLOCKCHAIN_PRIVATE_KEY = os.getenv(
    "BLOCKCHAIN_PRIVATE_KEY"
)

BLOCKCHAIN_CONTRACT_ADDRESS = os.getenv(
    "BLOCKCHAIN_CONTRACT_ADDRESS"
)

EXPECTED_CHAIN_ID = {
    "ganache": None,
    "sepolia": 11155111
}.get(BLOCKCHAIN_NETWORK)

if BLOCKCHAIN_NETWORK not in ("ganache", "sepolia"):
    raise ValueError(
        "BLOCKCHAIN_NETWORK must be 'ganache' or 'sepolia'."
    )

SOLC_VERSION = "0.8.20"
MODEL_VERSION = 7

app = Flask(__name__)
app.secret_key = os.environ.get(
    "MEDICHAIN_SECRET_KEY",
    "change-this-secret-before-production"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

logging.basicConfig(level=logging.INFO)
logger = app.logger


# ============================================================
# 2. FEATURES
# ============================================================

SYMPTOMS = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "headache": "Headache",
    "muscle_pain": "Muscle pain",
    "nausea": "Nausea",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "skin_rash": "Skin rash",
    "loss_smell": "Loss of smell",
    "loss_taste": "Loss of taste"
}

NUMERIC_FEATURES = [
    "age",
    *SYMPTOMS.keys(),
    "heart_rate",
    "temperature_c",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp"
]

FEATURES = ["gender"] + NUMERIC_FEATURES
TARGET = "diagnosis"

MEASUREMENT_RANGES = {
    "heart_rate": (40, 180),
    "temperature_c": (35, 41),
    "oxygen_saturation": (70, 100),
    "systolic_bp": (80, 200),
    "diastolic_bp": (40, 130)
}


# ============================================================
# 3. DATABASE
# ============================================================

def get_db():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                blockchain_hash TEXT NOT NULL,
                registration_tx TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                encrypted_data BLOB NOT NULL,
                record_hash TEXT NOT NULL,
                prediction TEXT NOT NULL,
                model_accuracy REAL,
                blockchain_tx TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS chain_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                contract_address TEXT NOT NULL
            );
        """)


# ============================================================
# 4. ENCRYPTION AND HASHING
# ============================================================

def get_cipher():
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
        logger.warning("Created a new encryption key.")

    return Fernet(KEY_PATH.read_bytes().strip())


def canonical_json(data):
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False
    )


def encrypt_record(data):
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = canonical_json(data).encode("utf-8")

    return get_cipher().encrypt(raw)


def decrypt_record(encrypted):
    if isinstance(encrypted, memoryview):
        encrypted = encrypted.tobytes()

    if isinstance(encrypted, str):
        encrypted = encrypted.encode("utf-8")

    plain = get_cipher().decrypt(bytes(encrypted))
    return json.loads(plain.decode("utf-8"))


def hash_record(data):
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = canonical_json(data).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ============================================================
# 5. BLOCKCHAIN
# ============================================================

CONTRACT_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MediChain {
    mapping(bytes32 => bool) public registeredUsers;
    mapping(uint256 => bytes32) public consultationHashes;

    function contractVersion() external pure returns (uint256) {
        return 1;
    }

    function registerUser(bytes32 userHash) external {
        require(!registeredUsers[userHash], "User already registered");
        registeredUsers[userHash] = true;
    }

    function registerConsultation(
        uint256 consultationId,
        bytes32 recordHash
    ) external {
        require(
            consultationHashes[consultationId] == bytes32(0),
            "Consultation already exists"
        );

        consultationHashes[consultationId] = recordHash;
    }

    function verifyUser(bytes32 userHash) external view returns (bool) {
        return registeredUsers[userHash];
    }

    function verifyConsultation(
        uint256 consultationId,
        bytes32 recordHash
    ) external view returns (bool) {
        return consultationHashes[consultationId] == recordHash;
    }
}
"""

_contract_interface = None


def blockchain_connection():

    web3 = Web3(
        Web3.HTTPProvider(
            BLOCKCHAIN_RPC_URL,
            request_kwargs={"timeout": 15}
        )
    )

    if not web3.is_connected():
        raise RuntimeError(
            f"Cannot connect to blockchain at {BLOCKCHAIN_RPC_URL}"
        )

    actual_chain_id = web3.eth.chain_id

    if (
        EXPECTED_CHAIN_ID is not None
        and actual_chain_id != EXPECTED_CHAIN_ID
    ):
        raise RuntimeError(
            f"Wrong blockchain network. "
            f"Expected chain ID {EXPECTED_CHAIN_ID}, "
            f"received {actual_chain_id}."
        )

    if BLOCKCHAIN_NETWORK == "ganache":
        if not web3.eth.accounts:
            raise RuntimeError(
                "Ganache returned no unlocked accounts."
            )

    elif BLOCKCHAIN_NETWORK == "sepolia":
        if not BLOCKCHAIN_PRIVATE_KEY:
            raise RuntimeError(
                "BLOCKCHAIN_PRIVATE_KEY is required for Sepolia."
            )

    return web3


def get_compiled_contract():
    global _contract_interface

    if _contract_interface is not None:
        return _contract_interface

    installed = {
        str(version) for version in get_installed_solc_versions()
    }

    if SOLC_VERSION not in installed:
        install_solc(SOLC_VERSION)

    compiled = compile_source(
        CONTRACT_SOURCE,
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
        evm_version="paris"
    )

    _contract_interface = next(iter(compiled.values()))
    return _contract_interface


def contract_is_compatible(web3, address, interface):
    try:
        if not web3.eth.get_code(address):
            return False

        contract = web3.eth.contract(
            address=address,
            abi=interface["abi"]
        )

        return contract.functions.contractVersion().call() == 1

    except Exception:
        return False


def get_contract():
    web3 = blockchain_connection()
    interface = get_compiled_contract()

    # --------------------------------------------------------
    # 1. Use explicitly configured contract address if provided
    # --------------------------------------------------------

    if BLOCKCHAIN_CONTRACT_ADDRESS:
        address = Web3.to_checksum_address(
            BLOCKCHAIN_CONTRACT_ADDRESS
        )

        if not contract_is_compatible(web3, address, interface):
            raise RuntimeError(
                "Configured smart contract is not compatible "
                "with the selected blockchain network."
            )

        return web3, web3.eth.contract(
            address=address,
            abi=interface["abi"]
        )

    # --------------------------------------------------------
    # 2. Sepolia must use an already-deployed contract
    # --------------------------------------------------------

    if BLOCKCHAIN_NETWORK == "sepolia":
        raise RuntimeError(
            "BLOCKCHAIN_CONTRACT_ADDRESS is required for Sepolia. "
            "Deploy the contract first and configure its address."
        )

    # --------------------------------------------------------
    # 3. Ganache: check the locally saved contract address
    # --------------------------------------------------------

    with get_db() as db:
        saved = db.execute(
            "SELECT contract_address FROM chain_config WHERE id = 1"
        ).fetchone()

    if saved:
        try:
            address = Web3.to_checksum_address(
                saved["contract_address"]
            )

            if contract_is_compatible(web3, address, interface):
                return web3, web3.eth.contract(
                    address=address,
                    abi=interface["abi"]
                )

        except Exception:
            logger.warning(
                "Saved Ganache contract is unavailable."
            )

    # --------------------------------------------------------
    # 4. Ganache only: deploy contract automatically
    # --------------------------------------------------------

    if BLOCKCHAIN_NETWORK != "ganache":
        raise RuntimeError(
            "Automatic contract deployment is only enabled for Ganache."
        )

    factory = web3.eth.contract(
        abi=interface["abi"],
        bytecode=interface["bin"]
    )

    tx_hash = factory.constructor().transact({
        "from": web3.eth.accounts[0],
        "gas": 3000000
    })

    receipt = web3.eth.wait_for_transaction_receipt(
        tx_hash,
        timeout=120
    )

    if receipt.status != 1:
        raise RuntimeError(
            "Smart contract deployment failed."
        )

    address = Web3.to_checksum_address(
        receipt.contractAddress
    )

    with get_db() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO chain_config(
                id,
                contract_address
            )
            VALUES (1, ?)
            """,
            (address,)
        )

    logger.info(
        "Ganache contract deployed at %s",
        address
    )

    return web3, web3.eth.contract(
        address=address,
        abi=interface["abi"]
    )



def send_contract_transaction(function):

    web3, _ = get_contract()

    private_key = BLOCKCHAIN_PRIVATE_KEY.strip()

    print("\n--- PRIVATE KEY DIAGNOSTICS ---")
    print("Loaded:", bool(private_key))
    print("Length before prefix removal:", len(private_key))
    print("Has 0x prefix:", private_key.startswith(("0x", "0X")))

    if private_key.startswith(("0x", "0X")):
        private_key = private_key[2:]

    print("Length after prefix removal:", len(private_key))
    print(
        "Only hexadecimal characters:",
        all(c in "0123456789abcdefABCDEF" for c in private_key)
    )
    print("--- END DIAGNOSTICS ---\n")

    # --------------------------------------------------------
    # GANACHE: use its unlocked development account
    # --------------------------------------------------------

    if BLOCKCHAIN_NETWORK == "ganache":

        tx_hash = function.transact({
            "from": web3.eth.accounts[0],
            "gas": 500000
        })

    # --------------------------------------------------------
    # SEPOLIA: sign transactions using the configured key
    # --------------------------------------------------------

    elif BLOCKCHAIN_NETWORK == "sepolia":

        if not BLOCKCHAIN_PRIVATE_KEY:
            raise RuntimeError(
                "Missing BLOCKCHAIN_PRIVATE_KEY in .env."
            )

        # Normalize the private key BEFORE using it.
        private_key = BLOCKCHAIN_PRIVATE_KEY.strip()

        if private_key.startswith(("0x", "0X")):
            private_key = private_key[2:]

        # Validate the private key without printing it.
        if len(private_key) != 64:
            raise ValueError(
                "Invalid Sepolia private key length. "
                "Expected 64 hexadecimal characters."
            )

        if not all(
            character in "0123456789abcdefABCDEF"
            for character in private_key
        ):
            raise ValueError(
                "Sepolia private key contains invalid characters. "
                "Check your .env configuration."
            )

        # Derive the account using the normalized key.
        account = web3.eth.account.from_key(private_key)

        # Confirm the account matches the wallet used to deploy
        # your MediChain contract.
        logger.info(
            "Sending Sepolia transaction from account: %s",
            account.address
        )

        transaction = function.build_transaction({
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(
                account.address,
                "pending"
            ),
            "chainId": web3.eth.chain_id,
            "gas": 500000,
            "gasPrice": web3.eth.gas_price
        })

        # Sign using the SAME normalized private key.
        signed_transaction = web3.eth.account.sign_transaction(
            transaction,
            private_key=private_key
        )

        raw_transaction = getattr(
            signed_transaction,
            "raw_transaction",
            None
        )

        if raw_transaction is None:
            raw_transaction = signed_transaction.rawTransaction

        tx_hash = web3.eth.send_raw_transaction(
            raw_transaction
        )

    else:
        raise RuntimeError(
            "Unsupported blockchain network."
        )

    # --------------------------------------------------------
    # WAIT FOR TRANSACTION CONFIRMATION
    # --------------------------------------------------------

    receipt = web3.eth.wait_for_transaction_receipt(
        tx_hash,
        timeout=120
    )

    if receipt.status != 1:
        raise RuntimeError(
            "Blockchain transaction failed."
        )

    return tx_hash.hex()



def register_user_on_chain(identity_hash):
    _, contract = get_contract()

    return send_contract_transaction(
        contract.functions.registerUser(
            bytes.fromhex(identity_hash)
        )
    )


def consultation_chain_id(consultation_id):
    value = str(consultation_id).strip()

    if value.isdigit():
        return int(value)

    return int(uuid.UUID(value).hex, 16)


def register_consultation_on_chain(consultation_id, record_hash):
    _, contract = get_contract()

    return send_contract_transaction(
        contract.functions.registerConsultation(
            consultation_chain_id(consultation_id),
            bytes.fromhex(record_hash)
        )
    )


def verify_consultation_on_chain(consultation_id, record_hash):
    _, contract = get_contract()

    return contract.functions.verifyConsultation(
        consultation_chain_id(consultation_id),
        bytes.fromhex(record_hash)
    ).call()


def recover_consultation_on_chain(consultation_id, record_hash):
    """
    Re-register an existing database record on the currently active
    Ganache contract if its ID is not already registered.

    This is a NEW registration on the current chain. It does not
    restore or prove an old transaction that disappeared after reset.
    """
    _, contract = get_contract()

    chain_id = consultation_chain_id(consultation_id)
    expected_hash = bytes.fromhex(record_hash)

    existing_hash = contract.functions.consultationHashes(
        chain_id
    ).call()

    if existing_hash == expected_hash:
        return None, True

    if existing_hash != bytes(32):
        raise RuntimeError(
            "This consultation ID already has a different hash "
            "on the current blockchain. It cannot be overwritten."
        )

    tx_hash = register_consultation_on_chain(
        consultation_id,
        record_hash
    )

    return tx_hash, True


# ============================================================
# 6. MACHINE LEARNING
# ============================================================

def dataset_fingerprint():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    stat = DATASET_PATH.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def make_pipeline(classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    gender_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("gender", gender_pipeline, ["gender"])
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred) * 100, 2
        ),
        "precision": round(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ) * 100, 2
        ),
        "recall": round(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ) * 100, 2
        ),
        "f1_score": round(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ) * 100, 2
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(y_true, y_pred) * 100, 2
        )
    }


def model_is_compatible(saved):
    required = {
        "models",
        "model",
        "model_name",
        "evaluation",
        "comparison",
        "version",
        "dataset_fingerprint"
    }

    return (
        isinstance(saved, dict)
        and required.issubset(saved.keys())
        and saved["version"] == MODEL_VERSION
        and saved["dataset_fingerprint"] == dataset_fingerprint()
    )


def train_model(force=False):
    if MODEL_PATH.exists() and not force:
        try:
            saved = joblib.load(MODEL_PATH)

            if model_is_compatible(saved):
                return saved

        except Exception:
            logger.exception("Saved model could not be loaded.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Put synthetic_medical_symptoms_dataset.csv "
            "inside the datasets folder."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + [TARGET]
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    df = df[required_columns].copy()
    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""]

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["gender"] = (
        df["gender"]
        .fillna("Other")
        .astype(str)
        .str.strip()
        .str.title()
    )

    counts = df[TARGET].value_counts()

    if len(counts) < 2:
        raise ValueError(
            "Dataset needs at least two diagnosis classes."
        )

    if counts.min() < 5:
        raise ValueError(
            "Each diagnosis needs at least 5 examples. "
            "More examples per diagnosis are recommended."
        )

    X = df[FEATURES]
    y = df[TARGET]

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=y_train_val
        )

    except ValueError as exc:
        raise ValueError(
            "Not enough examples in each diagnosis class to create "
            "stratified train, validation and test sets. "
            "Add more examples for each diagnosis."
        ) from exc

    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    validation_results = {}
    validation_scores = {}

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(X_train, y_train)

        validation_prediction = pipeline.predict(X_val)
        metrics = calculate_metrics(
            y_val,
            validation_prediction
        )

        validation_results[name] = metrics
        validation_scores[name] = metrics["f1_score"]

    best_name = max(
        validation_scores,
        key=validation_scores.get
    )

    # Refit all three models on the complete training/validation set.
    trained_models = {}
    comparison = []

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(X_train_val, y_train_val)

        trained_models[name] = pipeline

        test_prediction = pipeline.predict(X_test)
        metrics = calculate_metrics(
            y_test,
            test_prediction
        )

        comparison.append({
            "model": name,
            **metrics
        })

    best_model = trained_models[best_name]

    best_test_prediction = best_model.predict(X_test)
    best_test_probabilities = best_model.predict_proba(X_test)
    classes = best_model.classes_

    evaluation = calculate_metrics(
        y_test,
        best_test_prediction
    )

    evaluation["test_samples"] = int(len(y_test))
    evaluation["top_k"] = {}

    probability_order = np.argsort(
        best_test_probabilities,
        axis=1
    )[:, ::-1]

    for k in (1, 2, 3):
        effective_k = min(k, len(classes))

        hits = [
            y_test.iloc[index] in classes[
                probability_order[index, :effective_k]
            ]
            for index in range(len(y_test))
        ]

        evaluation["top_k"][f"top_{k}"] = round(
            float(np.mean(hits) * 100),
            2
        )

    matrix = confusion_matrix(
        y_test,
        best_test_prediction,
        labels=classes
    )

    evaluation["confusion_matrix"] = {
        "labels": [str(value) for value in classes],
        "matrix": matrix.tolist()
    }

    evaluation["classification_report"] = classification_report(
        y_test,
        best_test_prediction,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    saved = {
        "models": trained_models,
        "model": best_model,
        "model_name": best_name,
        "accuracy": evaluation["accuracy"],
        "evaluation": evaluation,
        "comparison": comparison,
        "validation_results": validation_results,
        "version": MODEL_VERSION,
        "dataset_fingerprint": dataset_fingerprint()
    }

    joblib.dump(saved, MODEL_PATH)

    logger.info(
        "All three models trained. Best validation model: %s",
        best_name
    )

    return saved


def predict_conditions(user_input):
    saved = train_model()

    frame = pd.DataFrame(
        [user_input],
        columns=FEATURES
    )

    all_predictions = {}
    model_confidence = []

    for model_name, model in saved["models"].items():
        probabilities = model.predict_proba(frame)[0]
        labels = model.classes_

        ranked = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        all_predictions[model_name] = [
            {
                "disease": str(label),
                "score": round(float(probability) * 100, 2),
                "probability": float(probability)
            }
            for label, probability in ranked[:3]
        ]

        model_confidence.append({
            "model": model_name,
            "disease": str(ranked[0][0]),
            "score": round(float(ranked[0][1]) * 100, 2),
            "probability": float(ranked[0][1])
        })

    best_name = saved["model_name"]
    best_predictions = all_predictions[best_name]

    highest_confidence = max(
        model_confidence,
        key=lambda item: item["score"]
    )

    return {
        "predictions": best_predictions,
        "all_predictions": all_predictions,
        "model_confidence": model_confidence,

        "model_name": best_name,
        "accuracy": saved["accuracy"],
        "evaluation": saved["evaluation"],
        "comparison": saved["comparison"],

        "highest_confidence_model": highest_confidence["model"],
        "highest_confidence_disease": highest_confidence["disease"],
        "highest_confidence_score": highest_confidence["score"]
    }


# ============================================================
# 7. INPUT VALIDATION
# ============================================================

def read_symptoms():
    try:
        age = int(request.form.get("age", ""))
    except (TypeError, ValueError):
        raise ValueError("Enter a valid age.")

    if not 1 <= age <= 110:
        raise ValueError("Age must be between 1 and 110.")

    gender = request.form.get(
        "gender",
        ""
    ).strip().title()

    if gender not in {"Male", "Female", "Other"}:
        raise ValueError("Select a valid gender.")

    data = {
        "age": age,
        "gender": gender
    }

    for key, label in SYMPTOMS.items():
        try:
            value = int(request.form.get(key, "0"))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid severity for {label}."
            )

        if value not in (0, 1, 2, 3):
            raise ValueError(
                f"{label} severity must be 0, 1, 2 or 3."
            )

        data[key] = value

    for field, (low, high) in MEASUREMENT_RANGES.items():
        raw = request.form.get(field, "").strip()

        if raw == "":
            data[field] = np.nan
            continue

        try:
            value = float(raw)
        except ValueError:
            raise ValueError(
                f"Enter a valid {field.replace('_', ' ')}."
            )

        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"{field.replace('_', ' ').title()} "
                f"must be between {low} and {high}."
            )

        data[field] = value

    return data


def clean_for_json(data):
    cleaned = {}

    for key, value in data.items():
        if isinstance(value, (float, np.floating)):
            cleaned[key] = (
                None if not np.isfinite(value)
                else float(value)
            )
        elif isinstance(value, (np.integer,)):
            cleaned[key] = int(value)
        else:
            cleaned[key] = value

    return cleaned


# ============================================================
# 8. LOGIN DECORATOR
# ============================================================

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# 9. PUBLIC ROUTES
# ============================================================

@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    return render_template("index.html", page="home")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("index.html", page="register")

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not all([name, email, username, password]):
        flash("Complete all registration fields.", "danger")
        return render_template("index.html", page="register")

    if len(password) < 8:
        flash("Password must contain at least 8 characters.", "danger")
        return render_template("index.html", page="register")

    with get_db() as db:
        duplicate = db.execute(
            """
            SELECT id FROM users
            WHERE username = ? OR email = ?
            """,
            (username, email)
        ).fetchone()

    if duplicate:
        flash("Username or email is already registered.", "danger")
        return render_template("index.html", page="register")

    try:
        identity_hash = hashlib.sha256(
            username.lower().encode("utf-8")
        ).hexdigest()

        registration_tx = register_user_on_chain(identity_hash)

        with get_db() as db:
            db.execute(
                """
                INSERT INTO users(
                    user_uuid, name, email, username,
                    password_hash, blockchain_hash, registration_tx
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    name,
                    email,
                    username,
                    generate_password_hash(password),
                    identity_hash,
                    registration_tx
                )
            )

                
            

        flash("Account created. Please sign in.", "success")
        return redirect(url_for("login"))

    except sqlite3.IntegrityError:
        flash("Username or email already exists.", "danger")

    except Exception:
        logger.exception("Registration failed.")
        flash(
            "Registration failed. Check that Ganache is running "
            "and inspect the VS Code terminal.",
            "danger"
        )

    return render_template("index.html", page="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("index.html", page="login")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    with get_db() as db:
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

    if (
        user is None
        or not check_password_hash(user["password_hash"], password)
    ):
        flash("Invalid username or password.", "danger")
        return render_template("index.html", page="login")

    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["name"] = user["name"]

    return redirect(url_for("dashboard"))


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have signed out.", "success")
    return redirect(url_for("index"))


# ============================================================
# 10. DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as db:
        user = db.execute(
            "SELECT * FROM users WHERE id = ?",
            (session["user_id"],)
        ).fetchone()

        total = db.execute(
            """
            SELECT COUNT(*) FROM consultations
            WHERE user_id = ?
            """,
            (session["user_id"],)
        ).fetchone()[0]

        recent = db.execute(
            """
            SELECT * FROM consultations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (session["user_id"],)
        ).fetchall()

    comparison = []
    best_model_name = None
    model_error = None

    try:
        saved = train_model()
        comparison = saved["comparison"]
        best_model_name = saved["model_name"]
    except Exception as exc:
        logger.exception("Dashboard model information unavailable.")
        model_error = str(exc)

    return render_template(
        "index.html",
        page="dashboard",
        user=user,
        total_consultations=total,
        recent_consultations=recent,
        comparison=comparison,
        best_model_name=best_model_name,
        model_error=model_error
    )


# ============================================================
# 11. SYMPTOM CHECKER
# ============================================================

@app.route("/symptom-checker", methods=["GET", "POST"])
@login_required
def symptom_checker():
    if request.method == "GET":
        return render_template(
            "index.html",
            page="symptom_checker",
            symptoms=SYMPTOMS
        )

    try:
        user_input = read_symptoms()
        result = predict_conditions(user_input)

        predictions = result["predictions"]
        all_predictions = result["all_predictions"]
        model_confidence = result["model_confidence"]

        model_name = result["model_name"]
        accuracy = result["accuracy"]
        evaluation = result["evaluation"]
        comparison = result["comparison"]

        highest_confidence_model = result["highest_confidence_model"]
        highest_confidence_disease = result["highest_confidence_disease"]
        highest_confidence_score = result["highest_confidence_score"]

        consultation_id = str(uuid.uuid4())

        medical_record = {
            "consultation_id": consultation_id,
            "user_id": session["user_id"],
            "input": clean_for_json(user_input),

            "predictions": predictions,
            "all_predictions": all_predictions,
            "model_confidence": model_confidence,

            "model_name": model_name,
            "model_accuracy": accuracy,
            "evaluation": evaluation,
            "comparison": comparison,

            "highest_confidence_model": highest_confidence_model,
            "highest_confidence_disease": highest_confidence_disease,
            "highest_confidence_score": highest_confidence_score,

            "created_at": datetime.now(timezone.utc).isoformat()
        }

        record_json = canonical_json(medical_record)
        encrypted_data = encrypt_record(record_json)
        record_hash = hash_record(record_json)

        # Submit to blockchain before saving the database row.
        blockchain_tx = register_consultation_on_chain(
            consultation_id,
            record_hash
        )

        with get_db() as db:
            db.execute(
                """
                INSERT INTO consultations(
                    consultation_id,
                    user_id,
                    encrypted_data,
                    record_hash,
                    prediction,
                    model_accuracy,
                    blockchain_tx
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    consultation_id,
                    session["user_id"],
                    encrypted_data,
                    record_hash,
                    predictions[0]["disease"],
                    accuracy,
                    blockchain_tx
                )
            )

        flash(
            "Assessment completed and record saved.",
            "success"
        )

        return render_template(
            "index.html",
            page="result",

            predictions=predictions,
            all_predictions=all_predictions,
            model_confidence=model_confidence,

            model_name=model_name,
            accuracy=accuracy,
            evaluation=evaluation,
            comparison=comparison,

            highest_confidence_model=highest_confidence_model,
            highest_confidence_disease=highest_confidence_disease,
            highest_confidence_score=highest_confidence_score,

            consultation_id=consultation_id,
            record_hash=record_hash,
            blockchain_tx=blockchain_tx
        )

    except ValueError as exc:

        logger.exception("ValueError in symptom checker")
        flash(str(exc), "danger")

    except Exception:
        logger.exception("Symptom checker failed.")
        flash(
            "Analysis could not be completed. Check the VS Code "
            "terminal for the error.",
            "danger"
        )

    return render_template(
        "index.html",
        page="symptom_checker",
        symptoms=SYMPTOMS
    )


# ============================================================
# 12. CONSULTATION HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():
    with get_db() as db:
        consultations = db.execute(
            """
            SELECT * FROM consultations
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (session["user_id"],)
        ).fetchall()

    return render_template(
        "index.html",
        page="history",
        consultations=consultations
    )


# ============================================================
# 13. CONSULTATION DETAILS
# ============================================================

@app.route("/consultation/<consultation_id>")
@login_required
def consultation_detail(consultation_id):
    with get_db() as db:
        consultation = db.execute(
            """
            SELECT * FROM consultations
            WHERE consultation_id = ? AND user_id = ?
            """,
            (consultation_id, session["user_id"])
        ).fetchone()

    if consultation is None:
        flash("Consultation not found.", "danger")
        return redirect(url_for("history"))

    try:
        medical_record = decrypt_record(
            consultation["encrypted_data"]
        )

        recalculated_hash = hash_record(medical_record)
        local_integrity = (
            recalculated_hash == consultation["record_hash"]
        )

        try:
            blockchain_verified = verify_consultation_on_chain(
                consultation_id,
                consultation["record_hash"]
            )
        except Exception:
            logger.exception("Blockchain detail verification failed.")
            blockchain_verified = False

        return render_template(
            "index.html",
            page="detail",
            consultation=consultation,
            medical_record=medical_record,
            local_integrity=local_integrity,
            blockchain_verified=blockchain_verified,
            verified=bool(local_integrity and blockchain_verified)
        )

    except InvalidToken:
        logger.exception("Invalid encryption key for consultation.")

        flash(
            "This record cannot be decrypted. Restore the original "
            "medichain.key file used when the record was created.",
            "danger"
        )

    except Exception:
        logger.exception(
            "Could not read consultation %s",
            consultation_id
        )
        flash("The consultation record could not be read.", "danger")

    return redirect(url_for("history"))


# ============================================================
# 14. BLOCKCHAIN VERIFICATION AND RECOVERY
# ============================================================

@app.route(
    "/verify/<consultation_id>",
    methods=["GET", "POST"]
)
@login_required
def verify_record(consultation_id):
    with get_db() as db:
        consultation = db.execute(
            """
            SELECT * FROM consultations
            WHERE consultation_id = ? AND user_id = ?
            """,
            (consultation_id, session["user_id"])
        ).fetchone()

    if consultation is None:
        flash("Consultation not found.", "danger")
        return redirect(url_for("history"))

    blockchain_verified = False
    local_integrity = None
    recovery_tx = None
    error_message = None

    # Verify local encrypted record and stored hash, where possible.
    try:
        record = decrypt_record(consultation["encrypted_data"])
        recalculated_hash = hash_record(record)

        local_integrity = (
            recalculated_hash == consultation["record_hash"]
        )

    except InvalidToken:
        local_integrity = False
        error_message = (
            "The record cannot be decrypted with the current "
            "medichain.key. Blockchain hash verification can still "
            "be attempted, but the medical details cannot be read."
        )

    except Exception:
        local_integrity = False
        logger.exception("Local record integrity check failed.")

    if request.method == "POST":
        if local_integrity is False:
            flash(
                "Recovery stopped: the local record could not be "
                "confirmed. Restore the correct encryption key "
                "before re-registering.",
                "danger"
            )
        else:
            try:
                recovery_tx, blockchain_verified = (
                    recover_consultation_on_chain(
                        consultation_id,
                        consultation["record_hash"]
                    )
                )

                if recovery_tx:
                    with get_db() as db:
                        db.execute(
                            """
                            UPDATE consultations
                            SET blockchain_tx = ?
                            WHERE consultation_id = ? AND user_id = ?
                            """,
                            (
                                recovery_tx,
                                consultation_id,
                                session["user_id"]
                            )
                        )

                    flash(
                        "Record hash registered on the current Ganache "
                        "contract. This is a new transaction.",
                        "success"
                    )
                else:
                    flash(
                        "This record hash is already registered "
                        "on the current blockchain.",
                        "success"
                    )

            except Exception as exc:
                logger.exception("Blockchain recovery failed.")
                error_message = str(exc)
                flash(
                    "Could not register this record on the current "
                    "blockchain. See the terminal for details.",
                    "danger"
                )

    if not blockchain_verified:
        try:
            blockchain_verified = verify_consultation_on_chain(
                consultation_id,
                consultation["record_hash"]
            )
        except Exception:
            logger.exception("Blockchain verification failed.")
            blockchain_verified = False

    return render_template(
        "index.html",
        page="verify",
        consultation=consultation,
        local_integrity=local_integrity,
        blockchain_verified=blockchain_verified,
        verified=bool(local_integrity and blockchain_verified),
        recovery_tx=recovery_tx,
        error_message=error_message
    )


# ============================================================
# 15. BLOCKCHAIN STATUS
# ============================================================

@app.route("/blockchain")
@login_required
def blockchain():
    connected = False
    latest_block = None
    network_id = None
    contract_address = None
    chain_error = None

    try:
        web3, contract = get_contract()

        connected = True
        latest_block = web3.eth.block_number
        network_id = web3.net.version
        contract_address = contract.address

    except Exception as exc:
        logger.exception("Blockchain status unavailable.")
        chain_error = str(exc)

    with get_db() as db:
        total_records = db.execute(
            """
            SELECT COUNT(*) FROM consultations
            WHERE user_id = ?
            """,
            (session["user_id"],)
        ).fetchone()[0]

    return render_template(
        "index.html",
        page="blockchain",
        blockchain_connected=connected,
        latest_block=latest_block,
        network_id=network_id,
        contract_address=contract_address,
        total_records=total_records,
        chain_error=chain_error
    )


# ============================================================
# 16. ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    flash("The requested page was not found.", "warning")

    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    return redirect(url_for("index"))


# ============================================================
# 17. START APPLICATION
# ============================================================

init_db()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )


