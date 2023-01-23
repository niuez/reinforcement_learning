#include <cmath>
#include <random>
#include <ostream>

struct State {
  double cart_pos;
  double cart_vel;
  double pole_pos;
  double pole_vel;
};

std::ostream& operator<<(std::ostream& os, const State& s) {
  return os << "{" << s.cart_pos
    << "," << s.cart_vel
    << "," << s.pole_pos
    << "," << s.pole_vel
    << "}";
}

struct CartPole_v1 {
  static constexpr double gravity = 9.8;
  static constexpr double masscart = 1.0;
  static constexpr double masspole = 0.1;
  static constexpr double total_mass = masspole + masscart;
  static constexpr double length = 0.5;  // actually half the pole's length;
  static constexpr double polemass_length = masspole * length;
  static constexpr double force_mag = 10.0;
  static constexpr double tau = 0.02;  // seconds between state updates;

  // Angle at which to fail the episode
  static constexpr double theta_threshold_radians = 12 * 2 * M_PI / 360;
  static constexpr double x_threshold = 2.4;

  State state;

  template<class URGB>
  void reset(URGB& g) {
    state.cart_pos = std::uniform_real_distribution<double>(-0.05, 0.05)(g);
    state.cart_vel = std::uniform_real_distribution<double>(-0.05, 0.05)(g);
    state.pole_pos = std::uniform_real_distribution<double>(-0.05, 0.05)(g);
    state.pole_vel = std::uniform_real_distribution<double>(-0.05, 0.05)(g);

  }

  bool step(int action) {
    double x = state.cart_pos;
    double x_dot = state.cart_vel;
    double theta = state.pole_pos;
    double theta_dot = state.pole_vel;
    double force = action == 1 ? force_mag : -force_mag;
    double costheta = std::cos(theta);
    double sintheta = std::sin(theta);
    double temp = (
            force + polemass_length * theta_dot * theta_dot * sintheta
        ) / total_mass;
    double thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
        );

    double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    x = x + tau * x_dot;
    x_dot = x_dot + tau * xacc;
    theta = theta + tau * theta_dot;
    theta_dot = theta_dot + tau * thetaacc;

    state = { x, x_dot, theta, theta_dot };

    bool terminated = 
            x < -x_threshold
            or x > x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians;
    return !terminated;
  }
};
