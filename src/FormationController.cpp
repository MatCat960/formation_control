#include "formation_control/FormationController.h"
#include "OsqpEigen/OsqpEigen.h"


#define M_PI   3.14159265358979323846  /*pi*/

namespace formation_control
{

    FormationController::FormationController(double fow_angle, double min_distance, double max_distance, int max_robots, int max_obstacles, int mates_num) : fow_angle_(fow_angle), min_distance_(min_distance), max_distance_(max_distance), max_robots_(max_robots), max_obstacles_(max_obstacles), mates_num_(mates_num)
    {
        std::cout << "Formation controller constructor called" << std::endl;
        H.resize(4, 4); // set sparse matrix size
        H.setZero();    // set all zeros
        f.resize(4);
        std::cout << "Hessian matrix initialized" << std::endl; 

        H.insert(0, 0) = 1.0;
        H.insert(1, 1) = 1.0;
        H.insert(2, 2) = 1.0;
        H.insert(3, 3) = 100.0;     // weight of slack var

        std::cout << "Hessian matrix filled" << std::endl;

        target_distance_ = max_distance_ * sin(M_PI/(mates_num_+1));           // corda di un cerchio = 2r*sin(th/2). r = R/2, th = 2*pi/(n+1)
        th_des = 2 * M_PI / (mates_num_ + 1);                                   // desired angle between mates to be evenly distributed on a circumference
        // target_distance_ = max_distance_ * sin(M_PI/);
        // A << tan(fow_angle_ / 2), 1.0, tan(fow_angle_ / 2), -1.0, 1.0, 0.0, -1.0, 0.0;

        // settings
        solver.settings()->setVerbosity(true);
        solver.settings()->setWarmStart(false);
        solver.settings()->setPolish(true);
        // solver.settings()->setPrimalInfeasibilityTollerance(1e-3);

        // set the initial data of the QP solver
        solver.data()->setNumberOfVariables(4);
        // Constraints: 
        // - 3 for max/min vel
        // - 1 slack var 
        // - 3 to keep target inside FoV 
        // - 1*N for safety distance from neighbors
        // - 1*N_obs for safety distance from obstacles
        // - 1*N_mates for CLF (desired formation) 
        // - 1*N_mates for max distance from mates (connectivity maintenance)
        vars_num_ = 4 + max_robots_ + max_obstacles_ + 2*mates_num_;             // DARS24
        // vars_num = 4 + max_robots_ + max_obstacles_ + mates_num_ + 1;
        solver.data()->setNumberOfConstraints(vars_num_);   
        std::cout << "Set number of variables and constraints" << std::endl;

        max_vel_ = 1.5;
        min_vel_ = -max_vel_;
        std::cout << "Set max vel" << std::endl;
        lowerbound.resize(vars_num_);
        lowerbound.setOnes();
        // lowerbound = -std::numeric_limits<double>::infinity() * lowerbound;
        lowerbound = -100000 * lowerbound;
        // lowerbound = -1.5 * lowerbound;

        upperbound.resize(vars_num_);
        upperbound.setOnes();
        // upperbound = std::numeric_limits<double>::infinity() * upperbound;
        upperbound = 1000000 * upperbound;
        // upperbound = 1.5 * upperbound;

        std::cout << "Set lower and upper bounds\n";


        // lowerbound.tail(3) = min_vel_ * Eigen::VectorXd::Ones(3);
        

        solver_init = false;
        std::cout << "Formation controller constructor finished" << std::endl;

    }

    FormationController::~FormationController()
    {
        solver.clearSolver();
    }

    void FormationController::setVelBounds(double v_min, double v_max)
    {
        max_vel_ = v_max;
        min_vel_ = v_min;
        lowerbound.head(3) = min_vel_ * Eigen::Vector3d::Ones();
        upperbound.head(3) = max_vel_ * Eigen::Vector3d::Ones();
    }

    void FormationController::setGamma(double gamma_fov, double gamma_safe, double gamma_clf)
    {
        gamma_fov_ = gamma_fov;
        gamma_safe_ = gamma_safe;
        gamma_clf_ = gamma_clf;
    }

    void FormationController::setVerbose(bool verbose)
    {
        solver.settings()->setVerbosity(verbose);
    }


    int FormationController::applyCbf(Eigen::Vector3d &uopt, Eigen::Vector3d &ustar, Eigen::MatrixXd &p_js_i, Eigen::Vector2d &p_t_i, Eigen::MatrixXd &obs_i, std::vector<Eigen::Vector2d> &mates, Eigen::VectorXd &h_out)
    {
        Eigen::SparseMatrix<double> A;
        A.resize(vars_num_, 4);
        
        // Constraints on min/max vel
        A.insert(0, 0) = 1;
        A.insert(1, 1) = 1;
        A.insert(2, 2) = 1;
        A.insert(3, 3) = 1;

        // Constraints on target
        /* --------- Removed for only coverage control (DARS24) ----------
        A.insert(4, 0) = tan(fow_angle_/2);
        A.insert(4, 1) = 1;
        A.insert(4, 2) = p_t_i(0) - p_t_i(1) * tan(fow_angle_/2);
        A.insert(4, 3) = -0.1;           // slack var
        A.insert(5, 0) = tan(fow_angle_/2);
        A.insert(5, 1) = -1;
        A.insert(5, 2) = -p_t_i(0) - p_t_i(1) * tan(fow_angle_/2);
        A.insert(5, 3) = -0.1;           // slack var
        A.insert(6, 0) = -2 * p_t_i(0);
        A.insert(6, 1) = -2 * p_t_i(1);
        A.insert(6, 2) = 0;
        A.insert(6, 3) = -0.1;           // slack var

        // std::cout << "A:\n" << A << std::endl;

        // Calculate h_fov
        Eigen::Vector3d h_fov;
        h_fov(0) = tan(fow_angle_/2) * p_t_i(0) + p_t_i(1);
        h_fov(1) = tan(fow_angle_/2) * p_t_i(0) - p_t_i(1);
        h_fov(2) = -pow(p_t_i.norm(), 2) + pow(max_distance_, 2);
        // std::cout << "Norm of target pos: " << p_t_i.norm() << std::endl;
        // std::cout << "Square of target pos: " << p_t_i.transpose() * p_t_i << std::endl;

        // std::cout << "h_fov: " << h_fov.transpose() << std::endl;

        upperbound(4) = gamma_fov_ * pow(h_fov(0),3);
        upperbound(5) = gamma_fov_ * pow(h_fov(1),3);
        upperbound(6) = gamma_fov_ * h_fov(2);
        // upperbound.block<1, 3>(3, 0) = gamma_fov_ * h_fov;

        // std::cout << "Upperbound: " << upperbound.transpose() << std::endl;
        -------------------------------------------------------------------------- */
        Eigen::Matrix<double, 4, 1> ustar_local = {ustar(0), ustar(1), ustar(2), 0.0};
        f = -ustar_local.transpose() * H;

        // CBF constraints (collision avoidance with other robots)
        for (int i = 0; i < max_robots_; i++)
        {
            Eigen::VectorXd p_j_i = p_js_i.col(i);
            // std::cout << "p_" << i << "_i: " << p_j_i.transpose() << std::endl;
            A.insert(4 + i, 0) = 2 * p_j_i(0);
            A.insert(4 + i, 1) = 2 * p_j_i(1);
            A.insert(4 + i, 2) = 0;
            A.insert(4 + i, 3) = -0.0001;           // slack var
            // A.insert(7 + i, 3) = -0.0001;           // slack var
            double h_i = pow(p_j_i.norm(), 2) - pow(min_distance_, 2);
            // std::cout << "h_" << i <<": " << h_i << std::endl;
            upperbound(4 + i) = gamma_safe_ * h_i;
        }

        // Obstacle avoidance constraints
        for (int i = 0; i < max_obstacles_; i++)
        {
            Eigen::VectorXd p_j_i = obs_i.col(i);
            A.insert(4 + max_robots_ + i, 0) = 2 * p_j_i(0);
            A.insert(4 + max_robots_ + i, 1) = 2 * p_j_i(1);
            A.insert(4 + max_robots_ + i, 2) = 0;
            A.insert(4 + max_robots_ + i, 3) = -0.0001;           // slack var
            double h_i = pow(p_j_i.norm(), 2) - pow(min_distance_, 2);
            upperbound(4 + max_robots_ + i) = gamma_safe_ * h_i;
        }

        // std::cout << "Defined CBF constraints" << std::endl;

        // Max distance from mates (connectivity maintenance)
        for (int i = 0; i < mates_num_; i++)
        {
            Eigen::VectorXd p_j_i = mates[i];
            A.insert(4 + max_robots_ + max_obstacles_ + i, 0) = -2 * p_j_i(0);
            A.insert(4 + max_robots_ + max_obstacles_ + i, 1) = -2 * p_j_i(1);
            A.insert(4 + max_robots_ + max_obstacles_ + i, 2) = 0;
            A.insert(4 + max_robots_ + max_obstacles_ + i, 3) = -0.0001;           // slack var
            double h_i = -pow(p_j_i.norm(), 2) + pow(max_distance_, 2);
            upperbound(4 + max_robots_ + max_obstacles_ + i) = gamma_fov_ * h_i;
        }

        // CLF constraints
        double V, K, z;
        // std::cout << "Desired distance: " << target_distance_ << std::endl;
        for (int i = 0; i < mates_num_; i++)
        {
            Eigen::VectorXd p_j_i = mates[i];
            z = p_j_i.norm();
            // std::cout << "Distance from robot " << i << ": " << z << std::endl;
            K = 2 * (z - target_distance_) / z;
            A.insert(4 + max_robots_ + max_obstacles_ + mates_num_ + i, 0) = -K * p_j_i(0);
            A.insert(4 + max_robots_ + max_obstacles_ + mates_num_ + i, 1) = -K * p_j_i(1);
            A.insert(4 + max_robots_ + max_obstacles_ + mates_num_ + i, 2) = 0;
            A.insert(4 + max_robots_ + max_obstacles_ + mates_num_ + i, 3) = -1.0;           // slack var
            V = pow(z - target_distance_, 2);
            upperbound(4 + max_robots_ + max_obstacles_ + mates_num_ + i) = -gamma_clf_ * V;
        }

        // std::cout << "Defined CLF constraints" << std::endl;

        // CLF on desired target position
        // double V, K, z;
        // z = p_t_i.norm();
        // K = 2 * (z - 0.5 * max_distance_) / z;
        // A.insert(4 + 3 + max_robots_ + max_obstacles_, 0) = -K * p_t_i(0);
        // A.insert(4 + 3 + max_robots_ + max_obstacles_, 1) = -K * p_t_i(1);
        // A.insert(4 + 3 + max_robots_ + max_obstacles_, 2) = 0;
        // A.insert(4 + 3 + max_robots_ + max_obstacles_, 3) = -1.0;           // slack var
        // V = pow(z - 0.5 * max_distance_, 2);
        // upperbound(4 + 3 + max_robots_ + max_obstacles_) = -gamma_clf_ * V;

        // z = p_t_i(1);
        // K = 2;
        // A.insert(4 + 3 + max_robots_ + max_obstacles_ + max_robots_ + 1, 0) = -K * p_t_i(0);
        // A.insert(4 + 3 + max_robots_ + max_obstacles_ + max_robots_ + 1, 1) = -K * p_t_i(1);
        // A.insert(4 + 3 + max_robots_ + max_obstacles_ + max_robots_ + 1, 2) = 0;
        // A.insert(4 + 3 + max_robots_ + max_obstacles_ + max_robots_ + 1, 3) = -0.1;           // slack var
        // V = pow(z, 2);
        // upperbound(4 + 3 + max_robots_ + max_obstacles_ + max_robots_ + 1) = -gamma_clf_ * V;


        

        // std::cout << "A : \n" << A << std::endl;


        if (!solver_init)
        {
            // set the initial data of the QP solver
            solver.data()->clearLinearConstraintsMatrix();
            solver.data()->clearHessianMatrix();
            if(!solver.data()->setHessianMatrix(H)) return 1;
            if (!solver.data()->setGradient(f)) return 1;
            if(!solver.data()->setLinearConstraintsMatrix(A)) return 1;
            if(!solver.data()->setLowerBound(lowerbound)) return 1;
            if(!solver.data()->setUpperBound(upperbound)) return 1;
            // instantiate the solver
            if(!solver.initSolver()) return 1;
            solver_init = true;
        } else
        {
            if (!solver.updateGradient(f)) return 1;
            if (!solver.updateLinearConstraintsMatrix(A)) return 1;
            if (!solver.updateUpperBound(upperbound)) return 1;
        }

        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;

        // get controller input
        auto u_i = solver.getSolution();
        uopt = u_i.head(3);
        auto hdot = A * u_i;

        h_out.resize(hdot.size());
        h_out = upperbound + hdot;

        // std::cout << "hdot: " << hdot.transpose() << std::endl;
        // std::cout << "lowerbound: " << lowerbound.transpose() << std::endl;
        // std::cout << "upperbound: " << upperbound.transpose() << std::endl;
        // std::cout << "Margin on obstacle avoidance constraints: " << (upperbound-hdot).transpose() << std::endl;

        // h_out = upperbound;

        return 0;
        

    }

}
