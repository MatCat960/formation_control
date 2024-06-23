#ifndef FORMATION_CONTROLLER_H
#define FORMATION_CONTROLLER_H

#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>



namespace formation_control
{
    class FormationController
    {
    private:
        double fow_angle_; //complete field of view angle of the camera
        double min_distance_; // minimum distance at which the drones should be kept at
        double max_distance_; // maximum distance at which the drones should be kept at
        double target_distance_; // desired distance among drones
        double th_des;
        Eigen::SparseMatrix<double> H; //hessian matrix of optimization problem
        Eigen::VectorXd f; // vector of optimization problem
        // Eigen::Matrix<double,4,2> A;
        // instantiate the solver
        OsqpEigen::Solver solver;
        Eigen::VectorXd lowerbound;
        Eigen::VectorXd upperbound;
        bool solver_init;
        int max_robots_;
        int max_obstacles_;
        int mates_num_;
        double max_vel_;
        double min_vel_;
        double gamma_fov_, gamma_safe_, gamma_clf_;
        int vars_num_;

    public:
        FormationController(double fow_angle, double min_distance, double max_distance, int max_robots_, int max_obstacles_, int mates_num);
        ~FormationController();
        int applyCbf(Eigen::Vector3d &uopt, Eigen::Vector3d &ustar, Eigen::MatrixXd &p_j_i, Eigen::Vector2d &p_t_i, Eigen::MatrixXd &obs_i, std::vector<Eigen::Vector2d> &mates, Eigen::VectorXd &h_out);
        // int applyCbfSingle(Eigen::Vector3d &uopt, Eigen::VectorXd &h_out, Eigen::Vector3d &ustar, Eigen::Vector3d &p_i, int n_robot, Eigen::MatrixXd &p_j, Eigen::MatrixXd slack);
        void setVerbose(bool verbose);
        void setVelBounds(double v_min, double v_max);
        void setGamma(double gamma_fov, double gamma_safe, double gamma_clf);
    };
}

#endif // FORMATION_CONTROLLER_H