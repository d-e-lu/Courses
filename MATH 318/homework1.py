import numpy as np
import matplotlib.pyplot as plt

def generalized_birthday(n, days):
    if n > days:
        return True
    days = np.random.choice(np.arange(1,days+1), size=n)
    days_seen = set()
    for day in days:
        if day in days_seen:
            return True
        else:
            days_seen.add(day)
    return False

def birthday(n):
    return generalized_birthday(n, 365)

def martian_birthday(n):
    return generalized_birthday(n, 669)

def generate_X(max, func):
    #Includes index 0 where n = 0
    X = [0] * (max + 1)
    for n in range(1, max+1):
        for i in range(1000):
            if func(n):
                X[n] += 1
        X[n] /= 1000.
    return X

def generate_Y(n, days):
    #Includes index 0 where n = 0
    Y = [0] * (n + 1)
    num = days
    curr = 1.
    for i in range(1, n+1):
        curr = curr * num / float(days)
        num -= 1
        Y[i] = 1 - curr
    return Y

def plot_it(X, Y, title):
    n = len(X) - 1
    fig, ax = plt.subplots()
    #don't include X[0] or Y[0]
    ax.plot(range(1, n+1), X[1:], label='1000 Sampled Points')
    ax.plot(range(1, n+1), Y[1:], label='True Probability')
    plt.ylabel('Probability')
    plt.xlabel('Number of people')
    plt.title(title)
    ax.legend()
    plt.show()

def plot_human_birthdays():
    n = 70
    X = generate_X(n, birthday)
    print("X[n] = ")
    print(X)

    Y = generate_Y(n, 365)
    print("Y[n] = ")
    print(Y)
    plot_it(X, Y, "Chance of Colliding Birthdays on Earth")

def plot_martian_birthdays():
    n= 70
    X_martian = generate_X(n, martian_birthday)
    print("X_martian[n] = ")
    print(X_martian)

    Y_martian = generate_Y(n, 669)
    print("Y_martian[n] = ")
    print(Y_martian)

    plot_it(X_martian, Y_martian, "Chance of Colliding Birthdays on Mars")


def main():
    plot_human_birthdays()
    plot_martian_birthdays()



if __name__ == "__main__":
    main()
