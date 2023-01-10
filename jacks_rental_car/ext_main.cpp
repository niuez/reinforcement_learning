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
  constexpr int Iteration = 50;
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
    Reward move_reward = (action <= -1 ? -action + 1 : std::abs(action)) * -2;
    i = std::min(stores[0].max_cars, i + action);
    j = std::min(stores[1].max_cars, j - action);
    Reward over_reward = (i > 10 ? -4 : 0) + (j > 10 ? -4 : 0);

    sigma += move_reward;
    sigma += over_reward;
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

/*
 415.27  425.25  435.11  444.72  453.97  462.78  471.14  479.04  486.89  494.35  501.71  508.70  515.21  521.65  527.07  531.53  534.10  538.83  543.11  547.04  550.69 
 425.13  435.11  444.97  454.58  463.83  472.64  480.99  488.89  496.35  503.71  510.70  517.21  523.65  529.07  533.53  537.16  540.83  545.11  549.04  552.69  556.13 
 434.55  444.52  454.38  463.99  473.23  482.04  490.38  498.27  505.71  512.70  519.21  525.65  531.07  535.53  539.16  542.83  547.11  551.04  554.69  558.13  561.42 
 443.23  453.20  463.06  472.66  481.90  490.68  499.00  506.86  514.26  521.20  527.65  533.07  537.53  541.16  544.83  549.11  553.04  556.69  560.13  563.42  566.49 
 451.09  461.06  470.91  480.50  489.71  498.46  506.73  514.54  521.87  528.73  535.07  539.53  543.16  546.83  551.11  555.04  558.69  562.13  565.42  568.49  571.29 
 458.20  468.17  478.01  487.58  496.74  505.43  513.63  521.35  528.59  535.33  541.53  545.16  548.21  552.74  556.84  560.62  564.13  567.42  570.49  573.29  575.74 
 464.91  474.71  484.53  494.06  503.16  511.76  519.85  527.45  534.55  541.13  547.16  550.08  553.43  557.76  561.67  565.27  568.63  571.78  574.72  577.39  579.72 
 472.50  481.71  490.74  500.11  509.12  517.60  525.54  532.96  539.88  546.27  552.08  554.39  557.91  562.01  565.72  569.14  572.34  575.34  578.13  580.66  582.87 
 479.71  488.74  497.43  505.83  514.73  523.04  530.78  537.99  544.68  550.82  556.39  558.13  561.71  565.58  569.08  572.31  575.33  578.17  580.81  583.20  585.28 
 486.74  495.43  503.76  511.85  520.03  528.15  535.66  542.60  549.00  554.85  560.13  561.36  564.89  568.52  571.79  574.82  577.66  580.34  582.82  585.06  587.01 
 493.43  501.76  509.85  517.54  525.04  532.94  540.18  546.81  552.89  558.41  563.36  563.72  567.53  570.89  573.94  576.76  579.42  581.92  584.24  586.34  588.16 
 499.76  507.85  515.54  522.96  529.99  536.68  543.00  548.89  554.41  559.36  562.15  563.53  566.89  569.94  572.76  575.42  577.92  580.24  582.34  584.16  584.87 
 505.60  513.54  520.96  527.99  534.68  541.00  546.89  552.41  557.36  560.31  564.59  564.89  567.94  570.76  573.42  575.92  578.24  580.34  582.16  583.76  585.34 
 511.04  518.78  525.99  532.68  539.00  544.89  550.41  555.36  558.24  562.76  566.76  566.70  569.26  571.91  574.30  576.53  578.65  580.66  582.54  584.24  585.72 
 516.15  523.66  530.60  537.00  542.89  548.41  553.36  555.89  560.67  564.93  568.70  568.44  570.80  573.26  575.48  577.55  579.51  581.38  583.13  584.73  586.12 
 520.94  528.18  534.81  540.89  546.41  551.36  553.09  558.24  562.79  566.85  570.44  569.99  572.21  574.53  576.60  578.53  580.36  582.11  583.75  585.25  586.56 
 521.40  528.35  534.65  540.37  545.54  550.31  555.32  560.28  564.65  568.55  571.99  571.36  573.51  575.69  577.65  579.46  581.18  582.82  584.37  585.78  587.03 
 525.49  532.14  538.11  543.48  548.31  552.76  557.27  562.06  566.28  570.05  573.36  572.53  574.65  576.73  578.58  580.29  581.92  583.48  584.95  586.29  587.48 
 529.16  535.53  541.19  546.24  550.76  554.93  558.97  563.61  567.69  571.33  574.53  573.48  575.62  577.61  579.38  581.01  582.57  584.05  585.46  586.75  587.88 
 532.39  538.51  543.89  548.67  552.93  556.85  560.55  564.90  568.86  572.38  575.48  574.20  576.40  578.32  580.01  581.58  583.08  584.51  585.87  587.11  588.21 
 535.18  541.09  546.24  550.79  554.85  558.55  562.05  565.89  569.75  573.18  576.20  574.79  576.97  578.83  580.48  582.00  583.46  584.85  586.17  587.39  588.46 
*/

/*
 0  0  0  0  0  0  0  0  1  1  2  2  2  3  4  5  4  4  4  4  4 
 0  0  0  0  0  0  0  0  0  1  1  1  2  3  4  5  3  3  3  3  4 
 0  0  0  0  0  0  0  0  0  0  0  1  2  3  4  2  2  2  2  3  3 
 0  0  0  0  0  0  0  0  0  0  0  1  2  3  1  1  1  1  2  2  2 
 0  0  0  0  0  0  0  0  0  0  0  1  2  0  0  0  0  1  1  1  1 
 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-2  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-3 -3 -2  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-4 -3 -3  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-4 -4 -3 -3  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -4 -4 -3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 
-5 -5 -4 -4 -3 -3 -2 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 
-5 -5 -5 -4 -4 -3 -2 -2 -2  0  0 -2 -2 -2 -2 -2 -2 -2 -2  0  0 
-5 -5 -5 -5 -4 -3 -3 -3  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -4 -4 -4  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -5  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4 -3  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4 -3  0  0  0  0  0  0  0  0  0  0  0  0  0  0 
*/
