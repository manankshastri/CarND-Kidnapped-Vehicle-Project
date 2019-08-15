/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

// using std::string;
// using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 15;  // TODO: Set the number of particles
  default_random_engine gen;

  // define normal distribution for the particle
  normal_distribution<double> norm_x(x, std[0]);
  normal_distribution<double> norm_y(y, std[1]);
  normal_distribution<double> norm_theta(theta, std[2]);

  for(int i=0; i<num_particles; i++){
    Particle newParticle;

    newParticle.id = i;
    newParticle.x = norm_x(gen);
    newParticle.y = norm_y(gen);
    newParticle.theta = norm_theta(gen);
    newParticle.weight = 1.0;

    particles.push_back(newParticle);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   default_random_engine gen;
   normal_distribution<double> noise_x(0.0, std_pos[0]);
   normal_distribution<double> noise_y(0.0, std_pos[1]);
   normal_distribution<double> noise_theta(0.0, std_pos[2]);

   for(int i=0;i<num_particles;i++){

     if(fabs(yaw_rate) > 0.00001){
       particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
       particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
       particles[i].theta += yaw_rate * delta_t;
     }
     else{
       particles[i].x += velocity * delta_t * cos(particles[i].theta);
       particles[i].y += velocity * delta_t * sin(particles[i].theta);
     }

     particles[i].x += noise_x(gen);
     particles[i].y += noise_y(gen);
     particles[i].theta += noise_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   // for(int i=0;i<observations.size(); i++){
   //   LandmarkObs obs = observations[i];
   //
   //   double min_dist = numeric_limits<double>::max();
   //
   //   int check_id = -1;
   //
   //   for(int j = 0;j<predicted.size(); j++){
   //     LandmarkObs pred = predicted[j];
   //
   //     double distance = dist(obs.x, obs.y, pred.x, pred.y);
   //
   //     if(distance < min_dist){
   //       min_dist = distance;
   //       check_id = pred.id;
   //     }
   //   }
   //
   //   observations[i].id = check_id;
   // }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
   double sig_X = 2.0 * std_landmark[0] * std_landmark[0];
   double sig_Y = 2.0 * std_landmark[1] * std_landmark[1];

   for(int i=0; i<num_particles; i++){
     Particle &p = particles[i];

     double totalWeight = 1.0;

     for(unsigned int j=0; j<observations.size(); j++){
       LandmarkObs obs = observations[j];

       double transformedX = p.x + (obs.x * cos(p.theta)) - (obs.y * sin(p.theta));
       double transformedY = p.y + (obs.x * sin(p.theta)) + (obs.y * cos(p.theta));

       double min_dist = numeric_limits<double>::max();
       int nearestLandmark = 0;

       for(unsigned int k=0; k<map_landmarks.landmark_list.size(); k++){

         double x_f = map_landmarks.landmark_list[k].x_f;
         double y_f = map_landmarks.landmark_list[k].y_f;

         double distance = dist(transformedX, transformedY, x_f, y_f);

         if(distance < min_dist){
           min_dist = distance;
           nearestLandmark = k;
         }
       }

       double exponent1 = (transformedX - map_landmarks.landmark_list[nearestLandmark].x_f) *
                          (transformedX - map_landmarks.landmark_list[nearestLandmark].x_f);
       double exponent2 = (transformedY - map_landmarks.landmark_list[nearestLandmark].y_f) *
                          (transformedY - map_landmarks.landmark_list[nearestLandmark].y_f);

       double exponent = (exponent1 / sig_X) + (exponent2 / sig_Y);

       double currentWeight = gauss_norm * exp(-exponent);

       totalWeight *= currentWeight;
     }
     weights[i] = totalWeight;
     p.weight = totalWeight;
   }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   default_random_engine gen;

   discrete_distribution<> distribution(weights.begin(), weights.end());

   vector<Particle> newParticles;

   for(int i=0; i<num_particles; i++){
     newParticles.push_back(particles[distribution(gen)]);
   }
   //particles = newParticles;
   particles = move(newParticles);  // suggestions
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
