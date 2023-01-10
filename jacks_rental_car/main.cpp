#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>

struct Store {
  int max_cars;
  double demand_mean;
  double supply_mean;
};

using Reward = double;

double poisson(double mean, int n) {
  return std::pow(mean, n) / std::tgamma(n + 1) * std::exp(-mean);
}

void policy_iteration(std::array<Store, 2> stores) {
  constexpr int Iteration = 20;
  constexpr double Gamma = 0.9;
  auto to_int = [&](int i, int j) { return (stores[1].max_cars + 1) * i + j; };

  int state_size = to_int(stores[0].max_cars, stores[1].max_cars) + 1;
  std::vector<Reward> value(state_size, 0);
  std::vector<int> pi(state_size, 0);

  std::array<std::vector<std::vector<double>>, 2> stores_coef;
  for(int i = 0; i < stores.size(); i++) {
    auto& coef = stores_coef[i];
    coef.resize(stores[i].max_cars + 1, std::vector<double>(stores[i].max_cars + 1, 0));
    for(int v = 0; v <= stores[i].max_cars; v++) {
      double d_sum = 0;
      for(int d = 0; d <= v; d++) {
        double p = d == v ? 1 - d_sum : poisson(stores[i].demand_mean, d);
        d_sum += p;
        double s_sum = 0;
        for(int s = 0; v - d + s <= stores[i].max_cars; s++) {
          double q = v - d + s == stores[i].max_cars ? 1 - s_sum : poisson(stores[i].supply_mean, s);
          s_sum += q;
          coef[v][v - d + s] += p * q;
        }
      }
    }
  }

  std::array<std::vector<double>, 2> rent_reward;
  for(int i = 0; i < stores.size(); i++) {
    auto& reward = rent_reward[i];
    reward.resize(stores[i].max_cars + 1, 0);
    for(int v = 0; v <= stores[i].max_cars; v++) {
      double d_sum = 0;
      for(int d = 0; d <= v; d++) {
        double p = d == v ? 1 - d_sum : poisson(stores[i].demand_mean, d);
        d_sum += p;
        reward[v] += p * d * 10;
      }
    }
  }


  auto calc_sigma = [&](int i, int j, int action) {
    Reward sigma = 0;
    Reward move_reward = std::abs(action) * -2;
    i = std::min(stores[0].max_cars, i + action);
    j = std::min(stores[1].max_cars, j - action);

    sigma += move_reward;
    sigma += rent_reward[0][i];
    sigma += rent_reward[1][j];
    for(int x = 0; x <= stores[0].max_cars; x++) {
      for(int y = 0; y <= stores[1].max_cars; y++) {
        sigma += stores_coef[0][i][x] * stores_coef[1][j][y] * Gamma * value[to_int(x, y)];
      }
    }
    return sigma;
  };


  for(int q = 0; q < Iteration; q++) {
    std::cerr << "iteration: " << q + 1 << std::endl;

    std::vector<Reward> next_value = value;

    for(int i = 0; i <= stores[0].max_cars; i++) {
      for(int j = 0; j <= stores[1].max_cars; j++) {
        next_value[to_int(i, j)] = calc_sigma(i, j, pi[to_int(i, j)]);
      }
    }

    value = std::move(next_value);

    for(int i = 0; i <= stores[0].max_cars; i++) {
      for(int j = 0; j <= stores[1].max_cars; j++) {
        double max_sigma = std::numeric_limits<Reward>::min();
        for(int action = std::max(-5, -i); action <= std::min(5, j); action++) {
          double sigma = calc_sigma(i, j, action);
          if(max_sigma < sigma) {
            pi[to_int(i, j)] = action;
            max_sigma = sigma;
          }
        }
      }
    }


    for(int i = 0; i <= stores[0].max_cars; i++) {
      for(int j = 0; j <= stores[1].max_cars; j++) {
        std::cerr << std::setw(7) << std::setprecision(2) << value[to_int(i, j)] << " ";
      }
      std::cerr << std::endl;
    }

    for(int i = 0; i <= stores[0].max_cars; i++) {
      for(int j = 0; j <= stores[1].max_cars; j++) {
        std::cerr << std::setw(2) << pi[to_int(i, j)] << " ";
      }
      std::cerr << std::endl;
    }
  }
}

int main() {
  std::array<Store, 2> stores;
  stores[0] = Store { 20, 3, 3 };
  stores[1] = Store { 20, 4, 2 };

  policy_iteration(stores);
}
