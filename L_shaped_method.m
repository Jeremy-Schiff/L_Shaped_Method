% Jeremy Schiff
% Description:
% Solves a two-stage linear program with recourse using the L-shaped method
% Idea is to try to solve it as a one stage program then iteratively add
% "feasibility" and "optimality" cuts to maximize the expected value of our
% behavior after the first stage. This is very useful to model outcomes
% with uncertainty wherein one needs to make a decision without perfect
% information.
%
% The form of the problem is given mathematically by:
%                 minimize c' * x + sum_K p(k) * q(k)' * y(k)
%                 such that Ax =< b; Tk * x + W * yk =< hk;
%
% Inputs for a problem with M x-coordinates, J x-only inequalities,
% P possible random outcomes, L y-coordinates, R mixed x-y inequalities
%      A: the coefficients on tne x_i inequalities, an M x J matrix
%      b: the bounds on the x_i inequalities, a J x 1 matrix
%      c: the weights given to x_i in minimization, an M x 1 matrix
%      T: the coefficients on x_i in the mixed inequalities, R x M x P dim
%      h: the bounds on the mixed x-y inequalities, an R x P matrix
%      W: the coefficients on y_i in the y_i inequalities, an L x R matrix
%      q: the weights given to y_i in minimization, an L x P matrix
%      p: the probabilities of the k_th random outcome, a P x 1 matrix
%      eps: the sensitivity for convergence, a float
%
% Returns:
%      x_optimal: the best J x 1 matrix solution for x
%      exit_flag: 1 if the value is trustworthy
%      num_...: represents the number of each type of iteration performed

function [x_optimal, exitflag, num_feasibility_cuts,  ...
                                num_optimality_cuts, num_iterations] ...
                             = L_shaped_method(A, b, c, T, q, h, W, p, eps)
num_feasibility_cuts = 0;
num_optimality_cuts = 0;
num_iterations = 0;
% For the first iteration where theta is undefined
c_null = [c, 0];
c_augmented = [c, 1];
constraints = A;
% Add in a coef on theta
constraints(:, length(c_augmented)) = 0;
bounds = b;
w_sz = size(W);
num_y_values = w_sz(2);
h_sz = size(h);
num_y_constraints = h_sz(1);
options = optimset('Display','none');

% Note for those who don't use MATLAB - return statements are abnormal, so
% there's little benefit to having a bool that changes upon completion.
% Instead, we just end the function at that one location.
while true
    % Solve the master problem    
    % We have no optimality constraints so theta is undefined
    if num_optimality_cuts == 0
    	[x_augmented, ~, exitflag, ~] = linprog(c_null, ...
                         constraints, bounds, [], [], [], [], [], options);
        if exitflag ~= 1
            return;
        end
        x_optimal = x_augmented(1 : end - 1);
        theta = -Inf;
    % We have an optimality constraint and can thus find theta
    else
    	[x_augmented, ~, exitflag, ~] = linprog(c_augmented,  ...
                         constraints, bounds, [], [], [], [], [], options);
        if exitflag ~= 1
            return;
        end
        x_optimal = x_augmented(1 : end - 1);
        theta = x_augmented(end);
    end
    num_iterations = num_iterations + 1;

    % Check if there is a need for feasibility cuts
    init_r = num_feasibility_cuts;
    for k = 1 : length(p)
        constraint_k = zeros(num_y_constraints, ... 
                                     num_y_values + 2 * num_y_constraints);
        % Produce our approximating equations to find violation
        for i = 1 : num_y_constraints
            constraint_k(i, 1 : num_y_values) = W(i, :);
            constraint_k(i, 2 * i - 1 + num_y_values) = 1;
            constraint_k(i, 2 * i + num_y_values) = -1;
        end
        f = ones(num_y_values + 2 * num_y_constraints, 1);
        % Only weigh the approximation coefs
        f(1 : num_y_values) = 0;
        lb = zeros(num_y_values + 2 * num_y_constraints, 1);
        lb(1 : num_y_values) = -Inf;
        [~, w_prime_test, exitflag, ~, duals] = linprog(f,  ...
                    constraint_k, h(:, k) - T(:, :, k) * x_optimal, [], ...
                                                  [], lb, [], [], options);
        if exitflag ~= 1
            return;
        end
        % See if we can meet our constraints
        if (w_prime_test > eps)
            % If not, calculate the cut using the LP dual
            coef = duals.ineqlin' * T(:, :, k);
            bound = duals.ineqlin' * h(:, k);
            % This is a little subtle: we need to keep any cut that is 
            % sufficiently large vs the bound, as well as any cut that 
            % contains sufficiently large  coefficients to allow for 
            % -3x_1 + 2x_2 <= 0 scenarios
            if any(abs(coef /(bound + max(coef))) > eps)
                coef = [coef, 0];
                constraints = [constraints; coef];
                bounds = [bounds; bound];
                num_feasibility_cuts = num_feasibility_cuts + 1;
            end
        end
    end
    %If our solution is feasible, we can perform an optimality cut
    if init_r == num_feasibility_cuts
        constraint_sum = zeros(1, num_y_values);
        bound_sum = 0;
        % Sum up the dual-derived optimality constraints for each
        % possibility
        for k = 1 : length(p)
            [~, ~, ~, ~, duals] = linprog(q(:, k), W, ...
                                     h(:, k) - T(:, :, k) * x_optimal,  ...
                                              [], [], [], [], [], options);
            bound_sum = bound_sum + p(k) * duals.ineqlin' * h(:, k);
            constraint_sum = constraint_sum + ... 
                                        p(k) * duals.ineqlin' * T(:, :, k);
        end
        % Create a cut from the optimality constraint we just found
        current_sum = constraint_sum * x_optimal - bound_sum;
        constraint = [constraint_sum, -1];
        constraints = [constraints; constraint];
        bounds = [bounds; bound_sum];
        % Check if we have reached our optimum vis-a-vis theta
        if eps >= (current_sum - theta) / abs(theta) ||  ...
                                                       current_sum <= theta
            return;
        else
            num_optimality_cuts = num_optimality_cuts + 1;
        end
    end
end