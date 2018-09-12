import sys
try:
    import docplex.mp
except: 
    if hasattr(sys, 'real_prefix'):
        get_ipython().system('pip install docplex')
    else:
        get_ipython().system('pip install --user docplex')
from docplex.mp.model import Model
import numpy as np
import timeit

#Reads the file and saves all the parameters is a list named parameters
def read_file(file_name):
    parameters=[]
    with open(file_name) as f:
        lines=f.read().splitlines()
        for line in lines:
            parameters.append([float(value) for value in line.split()])
    return parameters

#Constructs dictionaries to save all the parameters of the model
def set_parameters(parameters): 
    
    num_variables=int(parameters[0][0])
    num_constraints=int(parameters[0][1])
    num_cons_ineq=int(parameters[0][2])
    
    #Saves the cost coefficients in a dictionary
    costs_Objfunc1={}
    costs_Objfunc2={}
    costs_Objfunc3={}
    for i in range(num_variables):
        costs_Objfunc1.update({i:parameters[1][i]})
        costs_Objfunc2.update({i:parameters[2][i]})
        costs_Objfunc3.update({i:parameters[3][i]})
    
    #Saves the right hand side of the constraints in a dictionary
    right_hand={}
    for i in range(num_constraints):
        right_hand.update({i:parameters[4+num_constraints][i]})
    
   #Saves the constraint coefficients in a dictionary
    coef_const={}
    for i in range(4,num_constraints+4):
        for j in range(num_variables):
            coef_const.update({(i-4,j):parameters[i][j]})

    return num_variables, num_constraints, num_cons_ineq, costs_Objfunc1, costs_Objfunc2, costs_Objfunc3, right_hand, coef_const

#Constructs the model based on the parameters 
def construct_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const): 
    
    #Create one model instance
    m = Model()

    #Define the decision variables
    x={i:m.integer_var(name='x_{0}'.format(i)) for i in range(num_variables)}
    
    #Define the constraints
    #Constrainst with inequality 
    for i in range(num_cons_ineq):
        m.add_constraint( m.sum(x[j]*coef_const.get((i,j)) for j in range(num_variables))<= right_hand[i])

    #Constrainst with equality 
    for i in range(num_cons_ineq,num_constraints):
        m.add_constraint( m.sum(x[j]*coef_const.get((i,j)) for j in range(num_variables))== right_hand[i])
    
    return m, x

#Solves the model and returns the optimal solution and their image in the criterion space 
def solve_model(m, x, num_variables, costs_Objfunc): 
    
    #Solves the instance created under the name of m
    s = m.solve()
    
    #If the problem is infeasible
    if s is None:
        opt_solution=None
        Objfunc_value=None
    
    #If the problem is feasible
    else: 
        #Dictionary with the value of the optimal solution 
        opt_solution={}
        opt_solution.update({i:s[x[i]] for i in range(num_variables)})
    
        #Value of the objective function in the optimal solution
        Objfunc_value=np.sum(costs_Objfunc[i]*opt_solution[i] for i in range(num_variables))
    
    return opt_solution, Objfunc_value 

# Writes the output file
def write_output(nondominated_solutions, nondominated_points, elapsed, num_variables):
    
    #Creates the output file
    file = open("problem_solutions.txt", "w")
    #Writes the non dominated points, non dominated solutions and run time in the text file
    file.write("Nondominated points")
    for i in range(1,len(nondominated_points)+1): 
        file.write("\n"+str(i)+" Objective Function 1= "+str(nondominated_points[i-1][0])+" Objective Function 2= "+str(nondominated_points[i-1][1])+" Objective Function 3= "+str(nondominated_points[i-1][2]))
    
    file.write("\n"+ "Nondominated solutions")
    for i in range(1,len(nondominated_solutions)+1):
        file.write("\n"+str(i))
        for j in range(1,num_variables+1):
                   file.write(" X"+str(j)+"= "+str(nondominated_solutions[i-1][j-1]))
    
    file.write("\n"+"Run Time= "+str(elapsed)+" s")
                    
    file.close()

#Applies the two phase search method to find nondominated point in a rectangle
def two_phase_search(m,x,u,num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2, costs_Objfunc3):
    #Apply the 2 phase operation for a rectangle
    epsilon=0.99

    #PHASE 1. 
    # Adds the objective function 
    m.minimize(m.sum(x[i]*costs_Objfunc3.get((i))for i in range(num_variables))) 

    #Add contrainsts with the bound in the objective function value
    m.add_constraint(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))<= u[0]-epsilon, ctname="cons_OF1")
    m.add_constraint(m.sum(x[i]*costs_Objfunc2.get((i)) for i in range(num_variables))<= u[1]-epsilon, ctname="cons_OF2")

    #Solves the problem
    solution, z3= solve_model(m, x, num_variables, costs_Objfunc3)

    #Removes the 3rd Objective Function (OF3)
    m.remove_objective
    
    #Removes the bounding constrainst added for Objective Function 1 (OF1) and Objective Function 2 (OF2) in phase 1 
    m.remove_constraint("cons_OF1")
    m.remove_constraint("cons_OF2")

    #If the problem is infeasible
    if solution==None: 
        z2=None
        z1=None
       
    #If PHASE 1 is feasible conitnue to PHASE 2
    if solution !=None:
        #Calculates the value of the 1st and 2nd OF with the solution obtained before
        z2=np.sum(costs_Objfunc2[i]*solution[i] for i in range(num_variables))
        z1=np.sum(costs_Objfunc1[i]*solution[i] for i in range(num_variables))
    
        #PHASE 2
        #Adds the objective function adding OF1+OF2+OF3
        m.minimize(m.sum(x[i]*costs_Objfunc1.get((i))for i in range(num_variables))+m.sum(x[i]*costs_Objfunc2.get((i))for i in range(num_variables))+m.sum(x[i]*costs_Objfunc3.get((i))for i in range(num_variables)))

        #Add contrainsts with the bound in the objective function value
        m.add_constraint(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))<= z1+epsilon, ctname="cons_OF1")
        m.add_constraint(m.sum(x[i]*costs_Objfunc2.get((i)) for i in range(num_variables))<= z2+epsilon, ctname="cons_OF2")
        m.add_constraint(m.sum(x[i]*costs_Objfunc3.get((i)) for i in range(num_variables))<= z3+epsilon, ctname="cons_OF3")

        #Solves the problem
        solution, z3= solve_model(m, x, num_variables, costs_Objfunc3)
    
        #Calculates the value of the 1st and 2nd OF with the solution obtained before
        z2=np.sum(costs_Objfunc2[i]*solution[i] for i in range(num_variables))
        z1=np.sum(costs_Objfunc1[i]*solution[i] for i in range(num_variables))
    
        #Removes the Objective Function
        m.remove_objective
    
        #Removes the bounding constrainst added for Objective Function 1, Objective Function 2 and Objective Function 3 
        m.remove_constraint("cons_OF1")
        m.remove_constraint("cons_OF2")
        m.remove_constraint("cons_OF3")
    return z1, z2, z3, solution 

#Uses the basic quadrant shrinking method with the 2 phase method for rectangles to find non dominated points and solutions
def quadrant_shrinking(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const,costs_Objfunc1, costs_Objfunc2, costs_Objfunc3):
    epsilon=0.99

    # Constructs an object from the model
    m, x=construct_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const)

    #Creates a list of efficient solutions
    efficient_solutions=[]

    #Creates a list of efficient points
    efficient_points=[]

    #Initialize the double-ended linked list
    M=1000000
    double_link_list=[(M,M)]

    while len(double_link_list)!=0:
        right_boundary_not_treated=True
        while right_boundary_not_treated: 
            #Pop the front element of the double-ended link list
            u=double_link_list.pop(0)
        
            #Apply the 2 phase search method for rectangles
            z1, z2, z3, solution =two_phase_search(m,x,u,num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2, costs_Objfunc3)
        
            #If the problem is infeasible
            if solution==None:
                right_boundary_not_treated=False
            else:
                #Adds the efficient solution found
                efficient_solutions.append(solution)
                #Adds the efficient points found
                efficient_points.append({0:z1, 1:z2, 2:z3})
            
                #If the double link list is empty or the new u is not explored by other u´s in the double link list
                if(len(double_link_list)==0)or(double_link_list[0][0]<z1-epsilon):
                    #Add the new u to the front of the double ended link list
                    double_link_list.insert(0,(z1-epsilon,u[1]))
                #Add the other u to the front of the double ended link list
                double_link_list.insert(0,(u[0],z2-epsilon))
        
        top_boundary_not_treated=True
        while top_boundary_not_treated: 
            #Pop the back element of the double-ended link list
            u=double_link_list.pop(-1)
        
            #Apply the 2 phase search method for rectangles
            z1, z2, z3, solution =two_phase_search(m,x,u,num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2, costs_Objfunc3)
        
            #If the problem is infeasible
            if solution==None:
                top_boundary_not_treated=False
            else:
                #Adds the efficient solution found
                efficient_solutions.append(solution)
                #Adds the efficient points found
                efficient_points.append({0:z1, 1:z2, 2:z3})
            
                #If the double link list is empty or the new u is not explored by other u´s in the double link list
                if(len(double_link_list)==0)or(double_link_list[-1][1]<z2-epsilon):
                    #Add the new u to the back of the double ended link list
                    double_link_list.append((u[0],z2-epsilon))
                #Add the other u to the back of the double ended link list
                double_link_list.append((z1-epsilon,u[1]))
                
    return efficient_solutions, efficient_points


if __name__ == "__main__":
start_time = timeit.default_timer()
file_name='parameters.txt'
#Reads the input file
parameters= read_file(file_name)

#Sets the values of the parameters of the model
num_variables, num_constraints, num_cons_ineq, costs_Objfunc1, costs_Objfunc2,costs_Objfunc3, right_hand, coef_const=set_parameters(parameters)

#Applies the Basic Quadrant Shrinking Method
efficient_solutions, efficient_points=quadrant_shrinking(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const,costs_Objfunc1, costs_Objfunc2, costs_Objfunc3)

elapsed = timeit.default_timer() - start_time

#Writes de Output file
write_output(efficient_solutions, efficient_points, elapsed, num_variables)

