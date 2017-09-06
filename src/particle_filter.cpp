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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles);
	weights.resize(num_particles);

	for(int i=0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

		Particle p;
		{
			p.id=i;
			p.x = sample_x;
			p.y = sample_y;
			p.theta = sample_theta;
			p.weight = 1;
		}

		particles.push_back(p);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for(int i=0; i < num_particles; i++) {
		Particle& p = particles.at(i);

		p.x = p.x + velocity*( sin(p.theta+yaw_rate*delta_t) - sin(p.theta) )/yaw_rate;
		p.y = p.y + velocity*( cos(p.theta) - cos(p.theta+yaw_rate*delta_t) )/yaw_rate;
		p.theta = p.theta + yaw_rate*delta_t;

		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i=0; i < observations.size(); i++) {
		LandmarkObs& obs = observations.at(i);

		double min_dist = 0;
		for(int j=0; j < predicted.size(); j++) {
			LandmarkObs& pred = predicted.at(j);
			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			if (j == 0) {
				min_dist = distance;
			} else {
				if (distance < min_dist) {
					min_dist = distance;
					obs.id = pred.id;
				}
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sig_x = std_landmark[0], sig_y = std_landmark[1];
	double gauss_norm= 1/(2 * M_PI * sig_x * sig_y);

	for(int i=0; i < num_particles; i++) {
		Particle p = particles.at(i);

		vector<LandmarkObs> predicted;
		{
			for(int j=0; i < map_landmarks.landmark_list.size(); j++) {
				Map::single_landmark_s sls = map_landmarks.landmark_list.at(j);
				double distance = dist(sls.x_f, sls.y_f, p.x, p.y);

				if (distance <= sensor_range) {
					LandmarkObs lmo;
					lmo.id = sls.id_i;
					lmo.x = sls.x_f;
					lmo.y = sls.y_f;
					predicted.push_back(lmo);
				}
			}

			dataAssociation(predicted, observations);
		}

		{
			const int predicted_len = predicted.size();

			for(int k=0; k < observations.size(); k++) {
				LandmarkObs& lmo = observations.at(k);

				double x_obs, y_obs;
				{
					x_obs = p.x + lmo.x * cos(p.theta) - lmo.y * sin(p.theta);
					y_obs = p.y + lmo.y * sin(p.theta) + lmo.y * cos(p.theta);
				}

				if (predicted_len > 0 && lmo.id < predicted_len) {
					LandmarkObs& pred = predicted.at(lmo.id);
					double mu_x = pred.x, mu_y = pred.y;

					double exponent = (pow(x_obs - mu_x, 2))/(2 * pow(sig_x, 2)) + (pow(y_obs - mu_y, 2))/(2 * pow(sig_y, 2));

					p.weight *= gauss_norm * exp(-exponent);
				} else {
					p.weight = 0;
					break;
				}
			}

			weights.at(i) = p.weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> dd(weights.begin(), weights.end());

	vector<Particle> particles_resample;
	particles_resample.resize(num_particles);

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