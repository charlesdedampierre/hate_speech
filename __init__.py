import os
from dotenv import load_dotenv

load_dotenv()


RESULTS_PATH = os.getenv("RESULTS_PATH")

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

if not os.path.exists(RESULTS_PATH + "/agreement"):
    os.makedirs(RESULTS_PATH + "/agreement")

if not os.path.exists(RESULTS_PATH + "/agreement/kappa"):
    os.makedirs(RESULTS_PATH + "/agreement/kappa")

if not os.path.exists(RESULTS_PATH + "/agreement/percent"):
    os.makedirs(RESULTS_PATH + "/agreement/percent")
