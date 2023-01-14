#include "grid_wind_world.hpp"
#include <random>
#include <unordered_map>
#include <iostream>
#include <cassert>
#include <iomanip>

#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;

void Sarsa0_TD(const GridWindWorld& world) {
  constexpr double alpha = 0.5;
  constexpr double eps = 0.1;
  constexpr int Iteration = 1000000;
  constexpr double Gamma = 1;
  std::mt19937 mt(786);
  std::unordered_map<State, std::vector<double>> Q;
  std::unordered_map<State, int> max_pi;
  for(int x = 0; x < world.H; x++) {
    for(int y = 0; y < world.W; y++) {
      Q[{ x, y }] = std::vector<double>(4, 0);
      max_pi[{ x, y }] = 0;
    }
  }


  auto action_select = [&](const State& state) {
    static std::bernoulli_distribution eps_dist(1 - eps + eps / 4);
    static std::uniform_int_distribution<int> action_dist(0, 2);
    if(eps_dist(mt)) {
      return max_pi[state];
    }
    else {
      int action = action_dist(mt);
      if(max_pi[state] <= action) {
        action++;
      }
      return action;
    }
  };

  auto init_state = [&]() {
    static std::uniform_int_distribution<int> x_dist(0, world.H - 1);
    static std::uniform_int_distribution<int> y_dist(0, world.W - 1);
    return State { x_dist(mt), y_dist(mt) };
  };

  std::vector<double> iter;
  std::vector<double> loops;
  double action_cnt = 0;
  for(int q = 0; q < Iteration; q++) {
    State state = world.init_state();
    //State state = init_state();
    int action = action_select(state);
    while(!world.is_goal(state)) {
      action_cnt++;
      State next_state = state;
      double R = world.action(next_state, action);
      int next_action = action_select(next_state);
      Q[state][action] += alpha * (R + Gamma * Q[next_state][next_action] - Q[state][action]);
      max_pi[state] = std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
      state = std::move(next_state);
      action = next_action;
    }

    assert(world.is_goal(state));

    if((q + 1) % 1000 == 0) {
      iter.push_back(q + 1);
      loops.push_back(action_cnt / 1000.0);
      action_cnt = 0;
      std::cerr << "Iteration = " << q << " mean step =" << loops.back() << std::endl;
      const static std::vector<char> action_char = { '>', 'v', '<', '^' };
      for(int i = 0; i < world.H; i++) {
        for(int j = 0; j < world.W; j++) {
          int dir = max_pi[{ i, j }];
          std::cerr << action_char[dir];
        }
        std::cerr << std::endl;
      }
      std::cerr << "0001112210" << std::endl;
    }
  }

  std::vector<double> x, y, u, v;
  for(int i = 0; i < world.H; i++) {
    for(int j = 0; j < world.W; j++) {
      const static std::vector<int> dd = { 0, 1, 0, -1 };
      State s = { i, j };
      double MAX = *std::max_element(Q[s].begin(), Q[s].end());
      double MIN = *std::min_element(Q[s].begin(), Q[s].end());
      std::cerr << std::fixed << std::setprecision(10) << std::endl;
      std::cerr << i << " " << j << " " << MAX << " " << MIN << " = ";
      if(MAX == MIN) continue;
      for(int a = 0; a < 4; a++) {
        double len = 0.7 * (Q[s][a] - MIN) / (MAX - MIN);
        y.push_back(i);
        x.push_back(j);
        v.push_back(len * dd[a]);
        u.push_back(len * dd[a ^ 1]);
        std::cerr << " " << len;
      }
      std::cerr << std::endl;
    }
  }
  plt::quiver(x, y, u, v);
  //plt::quiver(x, y, u, v);
  plt::xlim(-1, 10);
  plt::ylim(-1, 7);
  //plt::xticks(std::vector<double>{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
  //plt::yticks(std::vector<double>{ -1, 0, 1, 2, 3, 4, 5, 6, 7 });
  plt::show();
}

int main() {
  GridWindWorld grid;
  grid.init_example();
  Sarsa0_TD(grid);
}

/*
>>>>v>>>>v
v>>>>>>>>v
v>>>>^>>>v
v>>>>>>>>v
v>>>>>>v<<
>>>>>>>>>>
>>>>>>v^^>
0001112210
*/
