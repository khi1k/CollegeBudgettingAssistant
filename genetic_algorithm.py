# Genetic algorithm for the College Budgeting Assistant.
# It evolves one monthly budget plan that the student follows every month of the semester.
# Every plan is repaired so it passes is_valid_budget before it gets a score.
# Scores use the same rewards as the Q-learning demo so the two can be compared fairly.

from math import inf
from random import Random

from constraint_satisfaction import generate_feasible_seed, is_valid_budget
from data_models import BudgetPlan, UserProfile
from reinforcement_learning import (
    DEFAULT_AID_REFUND,
    EMERGENCY_CHANCE,
    EMERGENCY_COST,
    RESERVE_GOAL,
    apply_month,
    budget_options,
    read_aid_refund,
    recommended_plan,
    train_agent,
)


POPULATION_SIZE = 40
GENERATIONS = 60
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3
TRAINING_SCENARIOS = 200
TEST_SCENARIOS = 1000


# ---------- Budget rules turned into limits for each category ----------

# Lists every category a plan should have: bills, lifestyle categories, personal spending and savings.
def categories_for(profile):
    categories = list(profile.fixed_expenses)
    for cat in profile.lifestyle_rules:
        if cat not in categories:
            categories.append(cat)
    for extra in ("personal", "savings"):
        if extra not in categories:
            categories.append(extra)
    return categories


# Works out the lowest and highest amount each category is allowed to get.
# Fixed bills are paid exactly (no reason to overpay), except savings, which can always grow.
def category_bounds(profile):
    bounds = {}
    for cat in categories_for(profile):
        rules = profile.lifestyle_rules.get(cat, {})
        low = max(profile.fixed_expenses.get(cat, 0.0), rules.get("min", 0.0))
        high = rules.get("max", inf)
        if cat in profile.fixed_expenses and cat != "savings":
            high = min(high, low)
        if low > high + 0.01:
            raise ValueError(f"Budget rules conflict: {cat} needs at least ${low:.2f} but is capped at ${high:.2f}")
        bounds[cat] = (low, max(low, high))
    return bounds


# Categories the algorithm is free to move money between.
def flexible_categories(bounds):
    return [cat for cat, (low, high) in bounds.items() if high - low > 0.01]


# ---------- Repair: makes any plan follow the budget rules ----------

# Pushes every category inside its limits, then makes the plan spend exactly the money available.
# Extra money goes to savings. If the plan spends too much, it trims categories that are above their minimum.
def repair(allocations, profile, bounds):
    fixed = {}
    for cat, (low, high) in bounds.items():
        fixed[cat] = min(max(allocations.get(cat, low), low), high)

    available = profile.total_available_monthly_cash
    gap = available - sum(fixed.values())

    if gap > 0:
        savings_room = bounds["savings"][1] - fixed["savings"]
        fixed["savings"] += min(gap, savings_room)
    elif gap < 0:
        slack = {cat: fixed[cat] - bounds[cat][0] for cat in fixed}
        total_slack = sum(slack.values())
        if total_slack > 0:
            cut = min(-gap, total_slack)
            for cat, extra in slack.items():
                fixed[cat] -= cut * extra / total_slack

    # Round to cents and put any rounding leftover into savings.
    fixed = {cat: round(amount, 2) for cat, amount in fixed.items()}
    leftover = round(available - sum(fixed.values()), 2)
    low, high = bounds["savings"]
    if low <= fixed["savings"] + leftover <= high:
        fixed["savings"] = round(fixed["savings"] + leftover, 2)

    return BudgetPlan(fixed)


# ---------- Fitness: how well a plan handles a semester of surprise bills ----------

# Builds a fixed list of practice semesters so every plan is scored on the same surprise bills.
def make_scenarios(profile, count, seed):
    rng = Random(seed)
    return [
        tuple(EMERGENCY_COST if rng.random() < EMERGENCY_CHANCE else 0 for _ in range(profile.semester_months))
        for _ in range(count)
    ]


# Plays out one semester. choose_plan(month, balance) returns the plan used that month.
# Same rewards as the Q-learning agent: points for spending, a bigger penalty for unpaid bills,
# and a bonus at the end for savings up to the emergency goal.
def semester_score(choose_plan, profile, emergencies):
    balance = profile.starting_cash
    score = 0.0
    for month, emergency in enumerate(emergencies, start=1):
        plan = choose_plan(month, balance)
        balance, shortfall = apply_month(balance, plan, emergency)
        score += 0.2 * plan.allocations.get("personal", 0) - 2 * shortfall
    return score + 0.5 * min(balance, RESERVE_GOAL)


# Average score of a plan over many practice semesters. Plans that break the rules score -inf.
def fitness(plan, profile, scenarios):
    valid, _ = is_valid_budget(plan, profile)
    if not valid:
        return -inf
    choose = lambda month, balance: plan
    average = sum(semester_score(choose, profile, s) for s in scenarios) / len(scenarios)
    # Tiny tie-breaker: when two plans score the same, prefer the one that saves the spare money
    # instead of putting it in a category like food above its minimum.
    return average + 0.001 * plan.allocations.get("savings", 0)


# ---------- Genetic operators ----------

# Starts with the safe seed plan and adds random ways to split the leftover money.
def initial_population(profile, bounds, size, rng):
    population = [generate_feasible_seed(profile)]
    population[0] = repair(population[0].allocations, profile, bounds)

    flexible = flexible_categories(bounds)
    minimums = {cat: low for cat, (low, _) in bounds.items()}
    leftover = profile.total_available_monthly_cash - sum(minimums.values())

    while len(population) < size:
        allocations = minimums.copy()
        weights = [rng.random() for _ in flexible]
        total_weight = sum(weights) or 1
        for cat, weight in zip(flexible, weights):
            allocations[cat] += leftover * weight / total_weight
        population.append(repair(allocations, profile, bounds))
    return population


# Picks the best of a few random plans to be a parent.
def tournament_select(population, scores, rng):
    picks = rng.sample(range(len(population)), min(TOURNAMENT_SIZE, len(population)))
    return population[max(picks, key=lambda i: scores[i])]


# Mixes two parent plans. Each category gets a blend of both parents' amounts.
def crossover(parent_a, parent_b, profile, bounds, rng):
    mix = rng.random()
    allocations = {
        cat: mix * parent_a.allocations.get(cat, 0) + (1 - mix) * parent_b.allocations.get(cat, 0)
        for cat in bounds
    }
    return repair(allocations, profile, bounds)


# Moves a random amount of money from one flexible category to another.
def mutate(plan, profile, bounds, rng):
    flexible = flexible_categories(bounds)
    if len(flexible) < 2:
        return plan
    minimums = sum(low for low, _ in bounds.values())
    step = max(1.0, 0.15 * (profile.total_available_monthly_cash - minimums))

    allocations = plan.allocations.copy()
    giver, taker = rng.sample(flexible, 2)
    amount = abs(rng.gauss(0, step))
    amount = min(amount, allocations[giver] - bounds[giver][0])
    allocations[giver] -= amount
    allocations[taker] += amount
    return repair(allocations, profile, bounds)


# ---------- Main loop ----------

# Runs the genetic algorithm and returns the best plan and the best score from each generation.
def evolve(profile, generations=GENERATIONS, population_size=POPULATION_SIZE, seed=441, report_every=10):
    rng = Random(seed)
    bounds = category_bounds(profile)
    scenarios = make_scenarios(profile, TRAINING_SCENARIOS, seed)

    population = initial_population(profile, bounds, population_size, rng)
    history = []

    for generation in range(generations + 1):
        scores = [fitness(plan, profile, scenarios) for plan in population]
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        history.append(scores[ranked[0]])

        if report_every and generation % report_every == 0:
            best = population[ranked[0]]
            print(f"Generation {generation:3d}  best score {scores[ranked[0]]:8.2f}  "
                  f"save {best.allocations['savings']:7.2f}  personal {best.allocations.get('personal', 0):6.2f}")

        if generation == generations:
            break

        # Keep the best plans unchanged, then fill the rest with children.
        next_population = [population[i] for i in ranked[:ELITE_COUNT]]
        while len(next_population) < population_size:
            parent_a = tournament_select(population, scores, rng)
            if rng.random() < CROSSOVER_RATE:
                parent_b = tournament_select(population, scores, rng)
                child = crossover(parent_a, parent_b, profile, bounds, rng)
            else:
                child = BudgetPlan(parent_a.allocations.copy())
            if rng.random() < MUTATION_RATE:
                child = mutate(child, profile, bounds, rng)
            next_population.append(child)
        population = next_population

    best = population[ranked[0]]
    valid, message = is_valid_budget(best, profile)
    if not valid:
        raise ValueError("Genetic algorithm produced an invalid plan: " + message)
    return best, history


# ---------- Demo ----------

# Same sample Abington student as the Q-learning demo.
def sample_profile(aid_refund=DEFAULT_AID_REFUND):
    semester_months = 4
    # Fall 2026 cost estimates for one semester at Abington.
    # https://tuition.psu.edu/rates-effective-2026-fall-semester-abington
    monthly_food = 3290 / semester_months
    monthly_transportation = 1122 / semester_months
    return UserProfile(
        monthly_income=1000,
        lump_sum_aid=aid_refund,
        semester_months=semester_months,
        starting_cash=0,
        fixed_expenses={"transportation": monthly_transportation},
        lifestyle_rules={"food": {"min": monthly_food}, "personal": {"max": 100}},
    )


def print_plan(plan):
    for cat, amount in plan.allocations.items():
        print(f"  {cat:15s} ${amount:8.2f}")
    print(f"  {'total':15s} ${plan.total_allocated:8.2f}")


# Compares the genetic algorithm with Q-learning and fixed savings on the same bills.
def demo(aid_refund=DEFAULT_AID_REFUND):
    profile = sample_profile(aid_refund)
    print("Penn State Abington genetic algorithm demo")
    print("Aid refund for the semester:", profile.lump_sum_aid)
    print("Monthly budget:", round(profile.total_available_monthly_cash, 2))

    try:
        generate_feasible_seed(profile)
    except ValueError as error:
        print(error)
        return

    print()
    best_plan, _ = evolve(profile)
    print()
    print("Best monthly plan found by the genetic algorithm")
    print_plan(best_plan)
    print("Passes budget rules:", is_valid_budget(best_plan, profile)[1])

    # Q-learning and fixed savings, built the same way as in reinforcement_learning.py.
    plans = budget_options(profile)
    q_values = train_agent(profile, plans)
    fixed_plan = min(plans, key=lambda plan: abs(plan.allocations["savings"] - 50))

    strategies = {
        "Genetic algorithm": lambda month, balance: best_plan,
        "Q-learning": lambda month, balance: recommended_plan(q_values, plans, month, balance),
        "Fixed savings": lambda month, balance: fixed_plan,
    }

    # New semesters the genetic algorithm never trained on, so the comparison is fair.
    test_scenarios = make_scenarios(profile, TEST_SCENARIOS, seed=2026)
    sample_bills = (0, 150, 0, 150)

    print()
    print(f"Average score over {TEST_SCENARIOS} new practice semesters (higher is better)")
    for label, choose in strategies.items():
        average = sum(semester_score(choose, profile, s) for s in test_scenarios) / len(test_scenarios)
        print(f"  {label:18s} {average:8.2f}")

    print()
    print("Same four months of sample bills:", sample_bills)
    for label, choose in strategies.items():
        balance = profile.starting_cash
        total_shortfall = 0
        personal_spending = 0
        for month, emergency in enumerate(sample_bills, start=1):
            plan = choose(month, balance)
            balance, shortfall = apply_month(balance, plan, emergency)
            total_shortfall += shortfall
            personal_spending += plan.allocations.get("personal", 0)
        print(f"  {label:18s} savings left {balance:8.2f}  personal spending {personal_spending:7.2f}  "
              f"unpaid bills {total_shortfall:7.2f}")


if __name__ == "__main__":
    demo(read_aid_refund())
