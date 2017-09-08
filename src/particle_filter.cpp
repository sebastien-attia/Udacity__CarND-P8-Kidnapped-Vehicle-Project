/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i < num_particles; i++) {
		Particle p;
		{
			p.id=i;
			p.x = dist_x(gen);
			p.y = dist_y(gen);
			p.theta = dist_theta(gen);
			p.weight = 1;
		}

		particles.push_back(p);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;

	for(int i=0; i < num_particles; i++) {
		Particle& p = particles.at(i);

		if (abs(yaw_rate) > 0.001) {
			p.x = p.x + velocity*( sin(p.theta+yaw_rate*delta_t) - sin(p.theta) )/yaw_rate;
			p.y = p.y + velocity*( cos(p.theta) - cos(p.theta+yaw_rate*delta_t) )/yaw_rate;
			p.theta = p.theta + yaw_rate*delta_t;
		} else {
			p.x = p.x + velocity*delta_t*cos(p.theta);
			p.y = p.y + velocity*delta_t*sin(p.theta);
		}

		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	const int predicted_len = predicted.size();

	for(int i=0; i < observations.size(); i++) {
		LandmarkObs& obs = observations.at(i);

		double min_dist = 0;
		for(int j=0; j < predicted_len; j++) {
			LandmarkObs& pred = predicted.at(j);
			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			if (j == 0) {
				min_dist = distance;
			} else {
				if (distance < min_dist) {
					min_dist = distance;
					obs.id = pred.id-1;
				}
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	double sig_x = std_landmark[0], sig_y = std_landmark[1];
	double gauss_norm= 1/(2 * M_PI * sig_x * sig_y);

	for(int i=0; i < num_particles; i++) {
		Particle& p = particles.at(i);

		vector<LandmarkObs> predicted;
		{
			for(int j=0; j < map_landmarks.landmark_list.size(); j++) {
				Map::single_landmark_s& sls = map_landmarks.landmark_list.at(j);

				LandmarkObs lmo;
				lmo.id = sls.id_i;
				lmo.x = sls.x_f;
				lmo.y = sls.y_f;
				predicted.push_back(lmo);
			}
		}

		vector<LandmarkObs> observations_map;
		{
			for(int k=0; k < observations.size(); k++) {
				LandmarkObs& lmo = observations.at(k);

				LandmarkObs trans_lmo;
				{
					trans_lmo.id = lmo.id;
					trans_lmo.x = p.x + lmo.x * cos(p.theta) - lmo.y * sin(p.theta);
					trans_lmo.y = p.y + lmo.x * sin(p.theta) + lmo.y * cos(p.theta);
					observations_map.push_back(trans_lmo);
				}
			}
		}

		dataAssociation(predicted, observations_map);

		{
			double prob = 1.0;
			for(int k=0; k < observations_map.size(); k++) {
				LandmarkObs& lmo = observations_map.at(k);
				LandmarkObs& pred = predicted.at(lmo.id);

				double distance = dist(pred.x, pred.y, p.x, p.y);

				if (distance > sensor_range)
					continue;

				double mu_x = pred.x, mu_y = pred.y;
				double exponent = pow(lmo.x - mu_x, 2)/(2 * pow(sig_x, 2)) + pow(lmo.y - mu_y, 2)/(2 * pow(sig_y, 2));

				prob *= gauss_norm * exp(-exponent);
			}

			p.weight = prob;
			weights.at(i) = p.weight;
		}
	}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	discrete_distribution<int> dd(weights.begin(), weights.end());

	vector<Particle> particles_resample;

	for(int i=0; i < num_particles; i++) {
		particles_resample.push_back( particles.at( dd(gen) ) );
	}

	particles = particles_resample;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
