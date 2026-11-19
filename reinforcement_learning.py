# Peer review demo. Abington's cost estimates are real. The student and bills are examples.

from collections import defaultdict
from math import isfinite
from random import Random

from constraint_satisfaction import generate_feasible_seed, is_valid_budget
from data_models import BudgetPlan, UserProfile


PERSONAL_SPENDING_TARGET = 100
EMERGENCY_COST = 150
EMERGENCY_CHANCE = 0.25
RESERVE_GOAL = 250
BALANCE_BUCKET = 50
DEFAULT_AID_REFUND = 800


# Try a few ways to split the leftover money and check each plan with the budget rules.
def budget_options(profile):
    seed = generate_feasible_seed(profile)
    savings = seed.allocations["savings"]
    savings_floor = max(
        profile.fixed_expenses.get("savings", 0),
        profile.lifestyle_rules.get("savings", {}).get("min", 0),
    )
    personal_spending = seed.allocations.get("personal", 0)
    available = min(
        max(0, savings - savings_floor),
        max(0, PERSONAL_SPENDING_TARGET - personal_spending),
    )
    plans = []
    for fraction in (0, 0.5, 1):
        shift = round(available * fraction, 2)
        allocations = seed.allocations.copy()
        allocations["personal"] = personal_spending + shift
        allocations["savings"] = savings - shift
        plan = BudgetPlan(allocations)
        valid, _ = is_valid_budget(plan, profile)
        if valid and all(plan.allocations != old.allocations for old in plans):
            plans.append(plan)

    if not plans:
        raise ValueError("None of the demo's spending options pass the budget rules.")
    return plans


# Group similar savings amounts so they can share a score for the same month.
def state_for(month, balance):
    return month, int(balance // BALANCE_BUCKET)


# Add the new savings and see how much of the surprise bill we can pay.
def apply_month(balance, plan, emergency):
    balance += plan.allocations["savings"]
    return max(0, balance - emergency), max(0, emergency - balance)


# Practice with random surprise bills and keep a score for each choice.
def train_agent(profile, plans, episodes=8000, seed=441):
    rng = Random(seed)
    q_values = defaultdict(lambda: [0.0] * len(plans))
    actions = range(len(plans))
    learning_rate = 0.1
    discount = 0.95
    exploration = 0.2

    for _ in range(episodes):
        balance = rng.choice((profile.starting_cash, RESERVE_GOAL))
        for month in range(1, profile.semester_months + 1):
            scores = q_values[state_for(month, balance)]
            if rng.random() < exploration:
                action = rng.choice(actions)
            else:
                action = scores.index(max(scores))

            emergency = EMERGENCY_COST if rng.random() < EMERGENCY_CHANCE else 0
            plan = plans[action]
            balance, shortfall = apply_month(balance, plan, emergency)
            # Give some points for spending money and take away more for unpaid bills.
            reward = 0.2 * plan.allocations["personal"] - 2 * shortfall
            target = reward
            if month == profile.semester_months:
                target += 0.5 * min(balance, RESERVE_GOAL)
            else:
                next_state = state_for(month + 1, balance)
                target += discount * max(q_values[next_state])

            scores[action] += learning_rate * (target - scores[action])

    return q_values


# Pick the plan with the best score for the student's situation.
def recommended_plan(q_values, plans, month, balance):
    scores = q_values[state_for(month, balance)]
    return plans[scores.index(max(scores))]


# Ask how much aid is left after the school bill is paid.
def read_aid_refund():
    prompt = "Enter the semester aid refund in dollars (Enter for " + str(DEFAULT_AID_REFUND) + "): "
    while True:
        try:
            response = input(prompt).strip()
        except EOFError:
            return DEFAULT_AID_REFUND

        if not response:
            return DEFAULT_AID_REFUND

        try:
            refund = float(response)
        except ValueError:
            print("Enter a number like 800 or 1200.50.")
            continue

        if isfinite(refund) and refund >= 0:
            return refund
        print("Enter a refund of zero or more.")


# Compare Q-learning and fixed savings for a sample Abington student.
def demo(aid_refund=DEFAULT_AID_REFUND):
    semester_months = 4

    # Fall 2026 cost estimates for one semester at Abington.
    # https://tuition.psu.edu/rates-effective-2026-fall-semester-abington
    monthly_food = 3290 / semester_months
    monthly_transportation = 1122 / semester_months

    # This student lives with family and pays no rent.
    profile = UserProfile(
        monthly_income=1000,
        lump_sum_aid=aid_refund,
        semester_months=semester_months,
        starting_cash=0,
        fixed_expenses={"transportation": monthly_transportation},
        lifestyle_rules={"food": {"min": monthly_food}, "personal": {"max": 100}},
    )
    monthly_leftover = profile.total_available_monthly_cash - (monthly_food + monthly_transportation)

    print("Penn State Abington peer review demo")
    print("Sample income and surprise bills")
    print("This student lives with family and pays no rent")
    print("Aid refund for the semester:", profile.lump_sum_aid)
    print("Aid set aside each month:", profile.monthly_aid_disbursement)
    print("Monthly budget:", profile.total_available_monthly_cash)
    print("Food per month (Abington estimate):", monthly_food)
    print("Transportation per month (Abington estimate):", monthly_transportation)
    if monthly_leftover < 0:
        print("Short each month after food and transportation:", round(-monthly_leftover, 4))
        return

    plans = budget_options(profile)
    q_values = train_agent(profile, plans)
    print("Left after food and transportation:", round(monthly_leftover, 4))
    print("Emergency savings goal:", RESERVE_GOAL)
    print()
    print("First month recommendations")
    for balance in (profile.starting_cash, RESERVE_GOAL):
        plan = recommended_plan(q_values, plans, 1, balance)
        print("Starting savings:", balance)
        print("Save:", plan.allocations["savings"], "Personal spending:", plan.allocations["personal"])

    # Give both plans the same four months of sample bills.
    emergencies = (0, 150, 0, 150)
    fixed_plan = min(plans, key=lambda plan: abs(plan.allocations["savings"] - 50))
    for label in ("Q-learning", "Fixed savings"):
        balance = profile.starting_cash
        total_shortfall = 0
        personal_spending = 0
        print()
        print(label)
        for month, emergency in enumerate(emergencies, start=1):
            if label == "Q-learning":
                plan = recommended_plan(q_values, plans, month, balance)
            else:
                plan = fixed_plan
            balance, shortfall = apply_month(balance, plan, emergency)
            total_shortfall += shortfall
            personal_spending += plan.allocations["personal"]
            print("Month", month, "Saved:", plan.allocations["savings"], "Bill:", emergency)
            print("Savings left:", balance, "Unpaid bill:", shortfall)
        print("Total personal spending:", personal_spending)
        print("Total unpaid bills:", total_shortfall)


if __name__ == "__main__":
    demo(read_aid_refund())
