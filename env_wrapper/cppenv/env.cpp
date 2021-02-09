#include "env.h"
#include <cmath>
#include <algorithm>

constexpr auto PI = (3.14159265358979323846);
constexpr auto EPS = (1e-8);
constexpr auto OBSTACLE_CNT = (40);
constexpr auto COLLISION_R = (30);
constexpr auto TARGET_R = (100);
constexpr auto UAV_COLLISION_R = (5);

inline double l2norm(double x, double y) 
{
	return sqrt(x * x + y * y);
}

inline double cross(double lx, double ly, double rx, double ry) 
{
	return lx * ry - rx * ly;
}

inline double dot(double lx, double ly, double rx, double ry) 
{
	return lx * rx + ly * ry;
}

UAVModel2D::UAVModel2D(double x, double y, double v, double w) : m_x(x), m_y(y), m_v(v), m_w(w)
{
}

void UAVModel2D::step(double ang, double acc)
{
	m_w += ang;
	m_v += acc;
	if (m_v > 10) m_v = 10;
	if (m_v < 0) m_v = 0;

	while (m_w > PI * 2) {
		m_w -= PI * 2;
	}
	while (m_w < -PI * 2) {
		m_w += PI * 2;
	}
	m_x += m_v * cos(m_w);
	m_y += m_v * sin(m_w);
}

ManyUavEnv::ManyUavEnv(int uav_cnt, int random_seed, int reward_type) : m_uav_cnt(uav_cnt), m_rnd_engine(random_seed), m_reward(reward_type)
{
	reset();
	m_collision.resize(OBSTACLE_CNT);
}

void ManyUavEnv::reset()
{
	m_uavs.clear();
	m_obstacles.clear();
	std::uniform_real_distribution<double> uav_dist_x(0, 2000), uav_dist_y(0, 400);
	for (int i = 0; i < m_uav_cnt; ++i) {
		m_uavs.push_back(UAVModel2D(
			uav_dist_x(m_rnd_engine),
			uav_dist_y(m_rnd_engine),
			0.0,
			PI / 2
		));
	}
	std::uniform_real_distribution<double> obs_dist_x(0, 2000), obs_dist_y(500, 1500);
	for (int i = 0; i < OBSTACLE_CNT; ++i) {
		m_obstacles.push_back({
			obs_dist_x(m_rnd_engine),
			obs_dist_y(m_rnd_engine)
		});
	}
	// initialize target position
	std::uniform_int_distribution<int> target_dist_x(500, 1500), target_dist_y(1600, 1800);
	m_target.x = target_dist_x(m_rnd_engine);
	m_target.y = target_dist_y(m_rnd_engine);
	m_steps = 0;
}

void ManyUavEnv::step(const ControlInfo& control)
{
	m_prev_pos.clear();
	m_next_pos.clear();
	for (int i = 0; i < m_uav_cnt; ++i) {
		m_prev_pos.push_back({ m_uavs[i].x(), m_uavs[i].y() });
		m_uavs[i].step(control[i][0], control[i][1]);
		m_next_pos.push_back({ m_uavs[i].x(), m_uavs[i].y() });
	}
	m_steps += 1;
}

std::vector<Observation> ManyUavEnv::getObservations() const
{
	std::vector<Observation> result;
	for (int i = 0; i < m_uav_cnt; ++i) {
		Observation obs;
		double cw = cos(m_uavs[i].w()), sw = sin(m_uavs[i].w());
		double uav_dir_x = cw, uav_dir_y = sw;
		double uav_right_x = uav_dir_y, uav_right_y = -uav_dir_x;
		// generate obstacle info
		for (int j = 8; j < 20; ++j) obs[j] = 2000.0;
		for (const auto& o : m_obstacles) {
			double obs_dir_x = o.x - m_uavs[i].x(), obs_dir_y = o.y - m_uavs[i].y();
			if (cross(obs_dir_x, obs_dir_y, uav_right_x, uav_right_y) >= 0) continue;
			double dist = l2norm(o.x - m_uavs[i].x(), o.y - m_uavs[i].y());
			double obs_dir_l = l2norm(obs_dir_x, obs_dir_y) + EPS;
			obs_dir_x /= obs_dir_l; obs_dir_y /= obs_dir_l;
			double theta = acos(dot(obs_dir_x, obs_dir_y, uav_right_x, uav_right_y));
			int index = static_cast<int>(floor(12 * theta / PI));
			if (index == 12) --index;
			if (dist < obs[index + 8]) obs[index + 8] = dist;
		}
		for (int j = 8; j < 20; ++j) obs[j] = (obs[j] - 1000.) / 2000.;
		// generate self info
		obs[0] = (m_uavs[i].x() - 1000.) / 2000.;
		obs[1] = (m_uavs[i].y() - 1000.) / 2000.;
		obs[2] = (m_uavs[i].v() - 5.) / 10.;
		obs[3] = (m_uavs[i].w() * 0.5) / PI;
		// generate closest uav info
		double min_dist = 20;
		int min_uav_index = -1;
		for (int j = 0; j < m_uav_cnt; ++j) {
			if (j == i) continue;
			double dist = l2norm(m_uavs[i].x() - m_uavs[j].x(), m_uavs[i].y() - m_uavs[j].y());
			if (dist < min_dist) {
				min_uav_index = j;
				min_dist = dist;
			}
		}
		if (min_uav_index != -1) {
			obs[4] = (m_uavs[min_uav_index].x() - 1000.) / 2000.;
			obs[5] = (m_uavs[min_uav_index].y() - 1000.) / 2000.;
			obs[6] = (m_uavs[min_uav_index].v() - 5.) / 10.;
			obs[7] = (m_uavs[min_uav_index].w() * 0.5) / PI;
		}
		else {
			obs[4] = -1.0;
			obs[5] = -1.0;
			obs[6] = 0.0;
			obs[7] = 0.0;
		}
		// generate target info
		obs[20] = (m_target.x - 1000.) / 500.;
		obs[21] = (m_target.y - 1700.) / 100.;
		result.push_back(obs);
	}
	return result;
}

inline double PLOG(double x) 
{
	return x < EPS ? 0 : log(x);
}

std::vector<double> ManyUavEnv::getRewards() const
{
	std::fill(m_collision.begin(), m_collision.end(), false);
	std::vector<double> result(m_uav_cnt, -3.0);

	for (int i = 0; i < m_uav_cnt; ++i) {
		double dist_prev = l2norm(m_prev_pos[i].x - m_target.x, m_prev_pos[i].y - m_target.y);
		double dist_next = l2norm(m_next_pos[i].x - m_target.x, m_next_pos[i].y - m_target.y);

        //distance based rule
        result[i] += tanh(0.2 * (10 - m_uavs[i].v())) * (dist_prev - dist_next); // original distance

		// check uav & obstacle collision
		bool collision = false;
		int obstacle_index = 0;
		for (const auto& o : m_obstacles) {
			if (l2norm(o.x - m_uavs[i].x(), o.y - m_uavs[i].y()) < COLLISION_R) {
				collision = true;
				m_collision[obstacle_index] = true;
				break;
			}
			++obstacle_index;
		}

		if (collision) result[i] -= 50.;
		// check uav & uav collision
		collision = false;
		for (int j = 0; j < m_uav_cnt; ++j) {
			if (i != j && l2norm(m_next_pos[j].x - m_next_pos[i].x, m_next_pos[j].y - m_next_pos[i].y) < UAV_COLLISION_R) {
				collision = true;
				break;
			}
		}
		if (collision) result[i] -= 50.;
		// formation in the circle
		if (l2norm(m_next_pos[i].x - m_target.x, m_next_pos[i].y - m_target.y) < TARGET_R) {
			result[i] += 5;
		}
	}
	return result;
}

std::vector<Point2D> ManyUavEnv::getObstacles() const
{
	return m_obstacles;
}

std::vector<Point2D> ManyUavEnv::getUavs() const
{
	return m_next_pos;
}

std::vector<bool> ManyUavEnv::getCollision() const
{
	return m_collision;
}

Point2D ManyUavEnv::getTarget() const
{
	return m_target;
}

bool ManyUavEnv::isDone() const
{
	return m_steps == 500;
}
