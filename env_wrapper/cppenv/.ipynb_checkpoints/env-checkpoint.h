#ifndef ENV_H_
#define ENV_H_

#include <random>
#include <vector>
#include <array>

using ControlInfo = std::vector<std::array<double, 2>>; // omega, acc
using Observation = std::array<double, 22>;


struct Point2D 
{
	double x, y;
};

class UAVModel2D
{
public:
	UAVModel2D(double x, double y, double v, double w);
	void step(double ang, double acc);

	double x() const { return m_x; }
	double y() const { return m_y; }
	double v() const { return m_v; }
	double w() const { return m_w; }

private:
	double m_x;
	double m_y;
	double m_v;
	double m_w;
};


class ManyUavEnv
{
public:
	ManyUavEnv(int uav_cnt, int random_seed, int reward_type);
	void reset();
	void step(const ControlInfo& control);
	std::vector<Observation> getObservations() const;
	std::vector<double> getRewards() const;

	std::vector<Point2D> getObstacles() const;
	std::vector<Point2D> getUavs() const;
	std::vector<bool> getCollision() const;
	Point2D getTarget() const;

	bool isDone() const;
private:
	int m_uav_cnt;
	std::default_random_engine m_rnd_engine;
	std::vector<UAVModel2D> m_uavs;
	std::vector<Point2D> m_prev_pos;
	std::vector<Point2D> m_next_pos;
	std::vector<Point2D> m_obstacles;
	Point2D m_target;

	mutable std::vector<bool> m_collision;
	int m_steps;

	//Point2D TARGET{ 1000, 1750 };
	int m_reward;
};


#endif // !ENV_H_
