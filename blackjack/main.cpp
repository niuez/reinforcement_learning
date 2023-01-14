#include "blackjack.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <cassert>
#include <iomanip>

#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;

void on_policy_eps_soft(const double eps) {
  constexpr int Iteration = 10000000;
  constexpr double Gamma = 1;
  std::mt19937 mt(786);
  std::unordered_map<State, int> max_pi;
  std::unordered_map<State, std::vector<double>> Q;
  std::unordered_map<State, std::vector<double>> N;

  for(int p_sum = 0; p_sum <= 21; p_sum++) {
    for(int p_ace = 0; p_ace <= 1; p_ace++) {
      if(p_sum == 21 && p_ace == 1) continue;
      for(int d = 1; d <= 10; d++) {
        Hand player = { p_sum, !!p_ace };
        Hand dealer = { d == 1 ? 0 : d, d == 1 };
        if(player.ace_update()) continue;
        int score = player.score();
        State state;
        state.player = player;
        state.dealer = dealer;
        max_pi[state] = score >= 20 || score == 0 ? 0 : 1;
        Q[state] = std::vector<double>(2, 0);
        N[state] = std::vector<double>(2, 0);

        //std::cerr << player.sum << " " << player.usable_ace << " " << dealer.sum << " " << dealer.usable_ace << std::endl;
      }
    }
  }
  for(int d = 1; d <= 10; d++) {
    Hand player;
    player.bust();
    Hand dealer = { d == 1 ? 0 : d, d == 1 };
    State state;
    state.player = player;
    state.dealer = dealer;
    max_pi[state] = 0;
    Q[state] = std::vector<double>(2, 0);
    N[state] = std::vector<double>(2, 0);
    //std::cerr << player.sum << " " << player.usable_ace << " " << dealer.sum << " " << dealer.usable_ace << std::endl;
  }

  std::bernoulli_distribution eps_dist(1 - eps + eps / 2);
  for(int q = 0; q < Iteration; q++) {
    //std::cerr << "----" << std::endl;
    State state;
    state.init(mt);
    std::vector<std::tuple<State, int, double>> hist;
    while(true) {
      //std::cerr << state.player.sum << " " << state.player.usable_ace << " " << state.player.score() << " " << state.dealer.sum << " " << state.dealer.usable_ace << std::endl;
      int action;
      if(state.is_bust()) {
        action = 0;
      }
      else if(eps_dist(mt)) {
        action = max_pi[state];
      }
      else {
        action = 1 - max_pi[state];
      }
      //std::cerr << action << std::endl;
      State before = state;
      auto [finished, R] = state.action(mt, action);
      //std::cerr << action << " " << finished << " " << R << std::endl;
      hist.push_back({ before, action, R });
      if(finished) break;
    }

    double G = 0;
    std::reverse(hist.begin(), hist.end());
    for(auto& [S, A, R]: hist) {
      auto& state = S;
      //std::cerr << state.player.sum << " " << state.player.usable_ace << " " << state.player.score() << " " << state.dealer.sum << " " << state.dealer.usable_ace << " " << A << " " << R << std::endl;
      G = Gamma * G + R;
      N[S][A] += 1;
      Q[S][A] = Q[S][A] + 1 / N[S][A] * (G - Q[S][A]);
      max_pi[S] = std::max_element(Q[S].begin(), Q[S].end()) - Q[S].begin();
    }

    if(q % 50000 == 0) {
      std::cerr << q << " " << Iteration << std::endl;
      std::map<std::tuple<int, int, int>, int> pi;
      for(auto& [s, a]: max_pi) {
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        pi[{ score, dealer, ace }] = a;
      }

      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << pi[{ i, d, ace }];
          }
          std::cerr << std::endl;
        }
      }

      std::map<std::tuple<int, int, int, int>, double> q;
      for(auto& [s, a]: Q) {
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        q[{ score, dealer, ace, 0 }] = a[0];
        q[{ score, dealer, ace, 1 }] = a[1];
      }


      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            /*
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{ i, d, ace, 0 }] << "/";
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{i, d, ace, 1}] << " ";
            */
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          }
          std::cerr << std::endl;
        }
      }
    }
  }

  {
    std::map<std::tuple<int, int, int, int>, double> q;
    for(auto& [s, a]: Q) {
      int score = s.player.score();
      int dealer = s.dealer.sum + s.dealer.usable_ace;
      bool ace = s.player.usable_ace;
      q[{ score, dealer, ace, 0 }] = a[0];
      q[{ score, dealer, ace, 1 }] = a[1];
    }


    for(int ace = 0; ace <= 1; ace++) {
      std::vector<std::vector<double>> x, y, z;
      std::vector<double> X, Y, Z;
      std::cerr << "ace = " << ace << std::endl;
      for(int i = 11; i <= 21; i++) {
        std::cerr << i << " : ";
        std::vector<double> ax, ay, az;
        for(int d = 1; d <= 10; d++) {
          /*
             std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
             std::cerr << q[{ i, d, ace, 0 }] << "/";
             std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
             std::cerr << q[{i, d, ace, 1}] << " ";
             */
          std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          ax.push_back(i);
          ay.push_back(d);
          az.push_back(q[{i, d, ace, 1}] - q[{i, d, ace, 0}]);
          X.push_back(i);
          Y.push_back(d);
          Z.push_back(q[{i, d, ace, 1}] - q[{i, d, ace, 0}]);
        }
        x.push_back(std::move(ax));
        y.push_back(std::move(ay));
        z.push_back(std::move(az));
        std::cerr << std::endl;
      }
      double M = 0;
      for(auto z: Z) {
        M = std::max(std::abs(z), M);
      }
      for(auto& z: Z) {
        z /= M;
      }
      plt::scatter_colored(Y, X, Z, 10.0, { { "vmin", "-1" }, {"vmax", "1"},  });
      //plt::plot_surface(x, y, z);
      plt::show();
      plt::clf();
    }
  }

}

void off_policy_mc(const double eps) {
  constexpr int Iteration = 3000000;
  constexpr double Gamma = 1;
  std::mt19937 mt(786);
  std::unordered_map<State, int> max_pi;
  std::unordered_map<State, std::vector<double>> Q;
  std::unordered_map<State, std::vector<double>> C;

  for(int p_sum = 0; p_sum <= 21; p_sum++) {
    for(int p_ace = 0; p_ace <= 1; p_ace++) {
      if(p_sum == 21 && p_ace == 1) continue;
      for(int d = 1; d <= 10; d++) {
        Hand player = { p_sum, !!p_ace };
        Hand dealer = { d == 1 ? 0 : d, d == 1 };
        if(player.ace_update()) continue;
        int score = player.score();
        State state;
        state.player = player;
        state.dealer = dealer;
        /*
        max_pi[state] = score >= 20 || score == 0 ? 0 : 1;
        Q[state] = std::vector<double>(2, max_pi[state] == 0 ? 0 : 1);
        C[state] = std::vector<double>(2, 0);
        */
        max_pi[state] = 0;
        Q[state] = std::vector<double>(2, 0);
        C[state] = std::vector<double>(2, 0);

        //std::cerr << player.sum << " " << player.usable_ace << " " << dealer.sum << " " << dealer.usable_ace << std::endl;
      }
    }
  }
  for(int d = 1; d <= 10; d++) {
    Hand player;
    player.bust();
    Hand dealer = { d == 1 ? 0 : d, d == 1 };
    State state;
    state.player = player;
    state.dealer = dealer;
    max_pi[state] = 0;
    Q[state] = std::vector<double>(2, 0);
    C[state] = std::vector<double>(2, 0);
    //std::cerr << player.sum << " " << player.usable_ace << " " << dealer.sum << " " << dealer.usable_ace << std::endl;
  }

  std::unordered_map<int, int> all_res;

  std::bernoulli_distribution eps_dist(1 - eps + eps / 2);
  for(int q = 0; q < Iteration; q++) {
    //std::cerr << "----" << std::endl;
    State state;
    state.init(mt);
    std::vector<std::tuple<State, int, double>> hist;
    while(true) {
      //std::cerr << state.player.sum << " " << state.player.usable_ace << " " << state.player.score() << " " << state.dealer.sum << " " << state.dealer.usable_ace << std::endl;
      int action;
      if(state.is_bust()) {
        action = 0;
      }
      else if(eps_dist(mt)) {
        action = max_pi[state];
      }
      else {
        action = 1 - max_pi[state];
      }
      //std::cerr << action << std::endl;
      State before = state;
      auto [finished, R] = state.action(mt, action);
      //std::cerr << action << " " << finished << " " << R << std::endl;
      hist.push_back({ before, action, R });
      if(finished) break;
    }

    double G = 0;
    double W = 1;
    std::reverse(hist.begin(), hist.end());

    all_res[std::get<2>(hist.front())]++;
    for(auto& [S, A, R]: hist) {
      //std::cerr << std::setprecision(5) << W << " ";
      G = Gamma * G + R;
      C[S][A] += W;
      Q[S][A] = Q[S][A] + W / C[S][A] * (G - Q[S][A]);
      double b = 0;
      if(S.is_bust()) {
        b = 1;
      }
      else if(max_pi[S] == A) {
        b = 1 - eps + eps / 2;
      }
      else {
        b = eps / 2;
      }
      W *= 1.0 / b;
      if(!S.is_bust()) {
        max_pi[S] = std::max_element(Q[S].begin(), Q[S].end()) - Q[S].begin();
      }
      if(max_pi[S] != A) {
        //std::cerr << "breaked";
        break;
      }
    }
    //std::cerr << std::endl;

    if(q % 50000 == 0) {
      std::cerr << q << " " << Iteration << std::endl;
      std::map<std::tuple<int, int, int>, int> pi;
      for(auto& [s, a]: max_pi) {
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        pi[{ score, dealer, ace }] = a;
      }

      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << pi[{ i, d, ace }];
          }
          std::cerr << std::endl;
        }
      }

      std::map<std::tuple<int, int, int, int>, double> q;
      for(auto& [s, a]: Q) {
        int score = s.player.score();
        int dealer = s.dealer.sum + s.dealer.usable_ace;
        bool ace = s.player.usable_ace;
        q[{ score, dealer, ace, 0 }] = a[0];
        q[{ score, dealer, ace, 1 }] = a[1];
      }


      for(int ace = 0; ace <= 1; ace++) {
        std::cerr << "ace = " << ace << std::endl;
        for(int i = 11; i <= 21; i++) {
          std::cerr << i << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{ i, d, ace, 0 }] << "/";
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{i, d, ace, 1}] << " ";
            //std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          }
          std::cerr << std::endl;
        }
        {
          std::cerr << "bs" << " : ";
          for(int d = 1; d <= 10; d++) {
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{ 0, d, ace, 0 }] << "/";
            std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
            std::cerr << q[{0, d, ace, 1}] << " ";
            //std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          }
          std::cerr << std::endl;
        }
      }
      std::cerr << all_res[-1] << "\t" << all_res[0] << "\t" << all_res[1] << std::endl;
    }
  }

  {
    std::map<std::tuple<int, int, int, int>, double> q;
    for(auto& [s, a]: Q) {
      int score = s.player.score();
      int dealer = s.dealer.sum + s.dealer.usable_ace;
      bool ace = s.player.usable_ace;
      q[{ score, dealer, ace, 0 }] = a[0];
      q[{ score, dealer, ace, 1 }] = a[1];
    }


    for(int ace = 0; ace <= 1; ace++) {
      std::vector<std::vector<double>> x, y, z;
      std::vector<double> X, Y, Z;
      std::cerr << "ace = " << ace << std::endl;
      for(int i = 11; i <= 21; i++) {
        std::cerr << i << " : ";
        std::vector<double> ax, ay, az;
        for(int d = 1; d <= 10; d++) {
          /*
             std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
             std::cerr << q[{ i, d, ace, 0 }] << "/";
             std::cerr << std::setprecision(1) << std::scientific << std::setw(8);
             std::cerr << q[{i, d, ace, 1}] << " ";
             */
          std::cerr << std::setprecision(1) << std::scientific << std::setw(8) << q[{i, d, ace, 1}] - q[{i, d, ace, 0}] << " ";
          ax.push_back(i);
          ay.push_back(d);
          az.push_back(q[{i, d, ace, 1}] - q[{i, d, ace, 0}]);
          X.push_back(i);
          Y.push_back(d);
          Z.push_back(q[{i, d, ace, 1}] - q[{i, d, ace, 0}]);
        }
        x.push_back(std::move(ax));
        y.push_back(std::move(ay));
        z.push_back(std::move(az));
        std::cerr << std::endl;
      }
      double M = 0;
      for(auto z: Z) {
        M = std::max(std::abs(z), M);
      }
      for(auto& z: Z) {
        z /= M;
      }
      plt::scatter_colored(Y, X, Z, 10.0, { { "vmin", "-1" }, {"vmax", "1"},  });
      //plt::plot_surface(x, y, z);
      plt::show();
      plt::clf();
    }
  }

}

int main() {
  //on_policy_eps_soft(0.1);
  off_policy_mc(0.3);
}

/*
ace = 0
     A23456789T
11 : 1111111111
12 : 1111111111
13 : 1111111111
14 : 1111111111
15 : 1110001111
16 : 1000001111
17 : 1000000100
18 : 0000000000
19 : 0000000000
20 : 0000000000
21 : 0000000000
ace = 1
     A23456789T
11 : 1111111111
12 : 1111111111
13 : 1111111111
14 : 1111111111
15 : 1111111111
16 : 1111111111
17 : 1111111111
18 : 1000000011
19 : 0000000000
20 : 0000000000
21 : 0000000000
*/
