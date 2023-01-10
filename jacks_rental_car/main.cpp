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

  auto calc_sigma = [&](int i, int j, int action) {
    Reward sigma = 0;
    Reward move_reward = std::abs(action) * -2;
    i = std::min(stores[0].max_cars, i + action);
    j = std::min(stores[1].max_cars, j - action);

    double dem_i = 0;
    for(int di = 0; di <= i; di++) {
      double pi = di == i ? 1 - dem_i : poisson(stores[0].demand_mean, di);
      dem_i += pi;

      double dem_j = 0;
      for(int dj = 0; dj <= j; dj++) {
        double pj = dj == j ? 1 - dem_j : poisson(stores[1].demand_mean, dj);
        dem_j += pj;

        double sup_i = 0;
        for(int si = 0; si + i - di <= stores[0].max_cars; si++) {
          double qi = si + i - di == stores[0].max_cars ? 1 - sup_i : poisson(stores[0].supply_mean, si);
          sup_i += qi;

          double sup_j = 0;
          for(int sj = 0; sj + j - dj <= stores[1].max_cars; sj++) {
            double qj = sj + j - dj == stores[1].max_cars ? 1 - sup_j : poisson(stores[1].supply_mean, sj);
            sup_j += qj;

            Reward rent_reward = (di + dj) * 10;
            Reward R = move_reward + rent_reward;
            double P = pi * pj * qi * qj;
            int next_i = si + i - di;
            int next_j = sj + j - dj;
            sigma += P * (R + Gamma * value[to_int(next_i, next_j)]);
          }
        }
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
