## AGENT-BASED RAMSEY GROWTH MODEL WITH BROWN AND GREEN CAPITALS (ABRam-BG)
## Date: 08.10.2025 
## Randomize initialization of farms' capitals


#Model design
import agentpy as ap
import numpy as np
import random

import networkx as nx

###########################################################################
#####           AGENTS
######################################################################

#WE DEFINE THE FARM AGENT
class Farm(ap.Agent):

###################################################
#
#       SETUP OF VARIABLES FOR THE AGENT
#
###################################################

    #Shared paremeters among farms
    alpha, delta, rho = None, None, None 

    def setup(self, alpha, delta, rho, beta, gamma, B0, G0):
    
        #Asigning value to the Farms' parameters
        Farm.alpha = alpha
        Farm.delta = delta
        Farm.rho = rho

        #(Initial) Capitals
        self.B = B0
        self.G = G0
        self.K = self.G + self.B

        #(Initial) Labor -> farms initially are endowed with one type of capital
        if self.B == 0 or self.G == 0:
            if self.B == 0:
                self.b = 0
                self.g = 1 
            else:
                self.b = 1 
                self.g = 0
        else:
            raise ValueError("Simulation aborted: Farm cannot hold both capitals for initializing.")
        
        self.L = self.b + self.g

        
        #(Initial) Production process 
        #(Initial) Technological progress levels
        self.beta = beta
        self.gamma = gamma
        #(Initial) Ratio between productivities
        self.tech_rat = self.gamma/self.beta
        #Farm's inital productions (Cobb-Douglas prod. Function)
        self.P_B = (self.B**Farm.alpha)*((self.beta*self.b)**(1-Farm.alpha))
        self.P_G = (self.G**Farm.alpha)*((self.gamma*self.g)**(1-Farm.alpha))
        self.P = self.P_B + self.P_G
        
        #(Initial) Investment (Investment of "ignorant" agents, in the type of capital they hold)
        if self.B == 0 or self.G == 0:
            if self.B == 0: 
                self.I_G = 0
                self.I_B = 0
            else:
                self.I_G = 0
                self.I_B = 0
        else:
            raise ValueError("Simulation aborted: Farm cannot hold both capitals for initializing.")

        #Counting process
        self.I = (Farm.alpha*Farm.rho*self.P - (1-delta)*self.K)/(1 + (Farm.alpha*Farm.rho))



        #(Initial) Consumption
        self.C = self.P - self.I

        #(Initial) Utilities u = instant utility, U = lifetime utility 
        self.u = np.log(self.C)
        self.U = self.u

        #(Initial) Environmental attitude -> farms' opinion about the environment
        # If they hold brown capital they don't care
        if self.G == 0:
            self.O = 0
        # If green capital they care
        else:
            self.O = 1
        
        self.new_O = self.O


###################################################
#
#             ACTIONS OF THE AGENT
#
###################################################  

    # Capital        
    def capital(self):
        self.B = (1 - Farm.delta)*self.B + self.I_B
        self.G = (1 - Farm.delta)*self.G + self.I_G

        self.K = self.B + self.G
        
    def read_news(self):
        # In this function the farm collects all the necessary information from the statistician
        
        #updates past information
        self.past_tech_rat = self.tech_rat
        
        #Current technologies
        self.beta = Statistician.real_beta
        self.gamma = Statistician.real_gamma

        
        #current productivity ratio
        if self.beta == 0:
            self.tech_rat = self.tech_rat
        else:
            self.tech_rat = self.gamma/self.beta
    
        
    # Labor allocation
    def labor(self):
        
        #Allocate labor according to capital holding
        if self.B == 0 or self.G ==0:
            if self.B == 0:
                #When farm holds only green capital
                self.b = 0
                self.g = 1 
            else:
                #When farm holds only brown capital 
                self.b = 1 
                self.g = 0
        else:
            #When farm holds two types of capital => optimize labor allocation
            tau = (self.tech_rat)**((1 - Farm.alpha)/Farm.alpha)
            capital_ratio = (self.G/self.B)

            self.b = 1/(capital_ratio*tau+1)  
            self.g = capital_ratio*tau*self.b
            
        self.L = self.b + self.g
                
    #Production
    def production(self):

        self.P_B = (self.B**Farm.alpha)*((self.beta*self.b)**(1-Farm.alpha))
        self.P_G = (self.G**Farm.alpha)*((self.gamma*self.g)**(1-Farm.alpha))

        #counting process
        self.P = self.P_B + self.P_G
        
# =================================================================================================
# ------------------             INVESTMENT PROCESS          -------------------------
# =================================================================================================
    
    #Investment
    def investment(self):
        
        #Investment depends on farms' opinion about the environment
        if self.O == 0: #i.e. I don't care about the environment
            self.I_G = 0 
            self.I_B = (Farm.alpha*Farm.rho*self.P_B - (1-Farm.delta)*self.B)/(1 + (Farm.alpha*Farm.rho))
        else: #I care about the environment
            self.I_G = (Farm.alpha*Farm.rho*self.P - (1-Farm.delta)*self.K)/(1 + (Farm.alpha*Farm.rho))
            self.I_B = 0
        
        #Counting process
        self.I = self.I_G + self.I_B

    #Consumption
    def consumption(self):
        self.C = self.P - self.I

    # Utility 
    def utility(self):
        self.u = np.log(self.C)
        self.U += self.u

    #Opinion process
    # Voter Dynamics
    def voter_interaction(self):
        if self.O == 1: # --> green agents are stubborn, they don't change opinion
            pass
        else:
            friends = self.network.neighbors(self).to_list() #we get a list of the farm's friends
            friend = random.choice(friends) #we randomly select a friend
            # print(f'farm {self.id} choosen friend farm {friend.id}')
            if friend.O != self.O:
                self.new_O = friend.O
                #print(f" ~Farm {self.id} going green! after talking with Farm {friend.id}")
            else:
                self.new_O = self.O

                        
    def update_opinion(self):
        #update farm's opinion after interaction
        self.O = self.new_O

#---------------------------------------------------------------------------------------------------------\
#---------------------------------------------------------------------------------------------------------

#STATISTICIAN
#We define the agent Statistician, she is going to compute all the aggregate variables of the system
class Statistician(ap.Agent):
    
    #SETUP OF VARIABLES FOR THE AGENT 
    def setup(self, beta, gamma):
        #The Statistician selects the simulation's farms
        self.farms = self.model.farms
        
        # CAPITALS
        Statistician.Total_B = sum(self.farms.B)
        Statistician.Total_G = sum(self.farms.G)
        Statistician.Total_K = Statistician.Total_B + Statistician.Total_G
        
        # Effective technical progress
        Statistician.real_beta = beta
        Statistician.real_gamma = gamma
        
        # GDP
        self.Total_P_G = sum(self.farms.P_G)
        self.Total_P_B = sum(self.farms.P_B)
        self.GDP = self.Total_P_G + self.Total_P_B
        
        # Total Investments
        self.Total_I_B = sum(self.farms.I_B)
        #Counting brown farms
        #Select brown farms
        brown_farms =  self.farms.select(self.farms.O == 0)
        self.B_farms = sum(1 for agent in brown_farms)
        
        #Select green farms
        self.Total_I_G = sum(self.farms.I_G)
        Statistician.Total_I = self.Total_I_B + self.Total_I_G

        #Consumption
        self.Total_C = sum(self.farms.C)

        # GDP growth rate
        self.g_rate = 0
        #Capital-to-output ratio
        self.K_output_rate = self.Total_K/self.GDP
        
        # Emissions DICE
        self.CO2_intensity = 0.291/(1000 * self.Total_P_B)
        self.C_CO2 = self.CO2_intensity*self.Total_P_B
        
        self.init_CO2 = self.CO2_intensity
        
        
###################################################
#
#         ACTIONS OF THE STATISTICIAN
#
###################################################  
        
        
    #Computes technical progresses of the economy at the time step
    def technical_progress(self):
        #To be able to recover the model of an economy with a single capital stock   
        if self.Total_G == 0 or self.Total_B == 0:
            if self.Total_G == 0:
                Statistician.real_beta = sum(self.farms.B)*(Statistician.real_beta/self.Total_B)
                Statistician.real_gamma = self.p.gamma # assume that technology does not change
            else:
                Statistician.real_beta = self.p.beta
                Statistician.real_gamma = sum(self.farms.G)*(Statistician.real_gamma/self.Total_G)  
       # There are two capital stocks in the economy
        else:
            Statistician.real_beta = sum(self.farms.B)*(Statistician.real_beta/self.Total_B)
            Statistician.real_gamma = sum(self.farms.G)*(Statistician.real_gamma/self.Total_G)
        
    #Compute aggregate capital of the economy at the time step   
    def aggregate_capital(self):
        Statistician.Total_B = sum(self.farms.B)
        Statistician.Total_G = sum(self.farms.G)
        self.Total_K = Statistician.Total_B + Statistician.Total_G
        
    #Count number of farms with brown opinion
    def count_farms_opinion(self):
        #To count brown farms
        brown_farms = self.farms.select(self.farms.O == 0)
        #Count brown farms
        self.B_farms = sum(1 for agent in brown_farms)
        
    #Computes fractional change between the current and a prior GDP
    #If you need the percentage change, multiply these values by 100.
    def growth_rate(self):
        self.g_rate = (sum(self.farms.P) - self.GDP)/self.GDP
        
    #Computes gross domestic product 
    def gross_domestic_product(self):
        self.Total_P_G = sum(self.farms.P_G)
        self.Total_P_B = sum(self.farms.P_B)
        self.GDP = sum(self.farms.P)
        
    #Computes gross investments
    def gross_investment(self):
        #Select brown farms
        self.Total_I_B = sum(self.farms.I_B)
        #Select green farms
        self.Total_I_G = sum(self.farms.I_G)
        
        Statistician.Total_I = sum(self.farms.I)

    #Computes the economy's aggregate consumption at the time step
    def gross_consumption(self):
        self.Total_C = sum(self.farms.C)
        
    #Computes the economy's capital-output ratio at the time step
    def capital_output_ratio(self):
        self.K_output_rate = self.Total_K/self.GDP
        
    #Computes the economy's emissions
    def count_emissions(self):
        t = self.p.time_steps_in_1_year
        self.CO2_intensity = self.CO2_intensity*(1 + (self.init_CO2*(1 - 0.015))**(1/t))
        
        self.C_CO2 = self.CO2_intensity*self.Total_P_B
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

###################################################
#
#                MODEL
#
################################################### 


class Economy(ap.Model):
    
    def setup(self):

        #Initialize the agents and network of the model. 
        #---------------------------------------------------------------
        #1. Initialize the model's networks
        #---------------------------------------------------------------
        # Network topology
        # Prepare a small-world network
        if self.p.network_topology == 'SW':
            net_seed = self.model.random.getrandbits(42) # Seed from model
            
            graph = nx.watts_strogatz_graph(
                    self.p.agents,
                    self.p.number_of_friends,
                    self.p.network_randomness,
                    seed=net_seed)
        
        # Or prepare a Fully connected newtwork
        elif self.p.network_topology == 'FC':
            graph = nx.complete_graph(self.p.agents)
        #---------------------------------------------------------------    
        #2. Creating farms
        #---------------------------------------------------------------
        
        ## 2.1. Randomize initial farm capitals (New Section)
        
        Farms = self.p.agents
        Tot_K0 = self.p.Tot_K0
        perc = self.p.green_perc
        
        #Compute total initial capital split
        Tot_G0 = perc*Tot_K0
        Tot_B0 = Tot_K0 - Tot_G0
        
        # Choose random subset of farms to be initially green
        green_farms = int(perc * Farms)
        green_ids = self.random.sample(range(Farms), green_farms)

        # Allocate capital vectors
        B0 = np.zeros(Farms)
        G0 = np.zeros(Farms)

        for i in range(Farms):
            if i in green_ids:
                G0[i] = Tot_G0 / green_farms
            else:
                B0[i] = Tot_B0 / (Farms - green_farms)
                
        # Compute initial technical progress
        #Parameter such that farms investment is bigger that its capital depreciation, i.e I > delta K
        alpha, delta, rho = self.p.alpha, self.p.delta, self.p.rho
        Omega = (delta + (1/(rho*alpha)))**(1/(1 - alpha))
        gamma0 = Omega * G0.max() + 1
        beta0 = gamma0 / perc
                
        ## 2.2 Create farms with randomized initial values
        
        self.farms = ap.AgentList(self, self.p['agents'], Farm, alpha = self.p['alpha'], 
                                   delta = self.p['delta'], rho = self.p['rho'],
                                   beta=beta0, gamma=gamma0,
                                   G0 = ap.AttrIter(G0),
                                   B0 = ap.AttrIter(B0) )
        # ---------------------------------------------------------------
        # 2.3. Record for diagnostics
        # ---------------------------------------------------------------
        self.record('initial_green_fraction', perc)
        self.record('initial_green_ids', green_ids)
        self.record('Tot_G0', Tot_G0)
        self.record('Tot_B0', Tot_B0)
        self.record('gamma0', gamma0)
        self.record('beta0', beta0)
        
        
        #---------------------------------------------------------------
        #3. Creating statistician
        #---------------------------------------------------------------
        self.statistician = ap.AgentList(self, 1, Statistician, beta=beta0, gamma=gamma0,)
        
        #3.1 Setting a variable for all the model's agents
        self.all_agents = self.farms + self.statistician
        #---------------------------------------------------------------
        #4. Embedding agents into network
        #---------------------------------------------------------------
        self.network = self.farms.network = ap.Network(self, graph)
        self.network.add_agents(self.farms, self.network.nodes)
        
        #Optional: Plot initial network
        if self.p.draw_network == True:
            fig, ax = plt.subplots(figsize=( 215/25.4, 180/(2*25.4),))
            color_dict = {0:'#744700', 1:'#008000'}
            colors = [color_dict[c] for c in self.farms.O]
            nx.draw_circular(self.network.graph, node_color=colors,
                         node_size=50)
        else:
            pass
        
    def step(self):
                
        #FARMS interaction
        if self.p.interaction == True:
            if self.p.opinion_dynamics == True:
                #Then, we choose what type of interaction
                #VOTER DYNAMICS
                if self.p.interaction_type == 'Voter':
                    if self.p.interacting_farms == self.p.agents: 
                        #All the farms interact at each time step
                        self.farms.voter_interaction()
                    else:
                        #Otherwise draw certain number of random agents to interact
                        rand_agents = self.farms.random(n=self.p.interacting_farms,replace=True) 
                        for i, agent in enumerate(rand_agents):
                            agent.voter_interaction()
                #MAJORITY RULE
                elif self.p.interaction_type == 'MajorityRule': 
                    #Draw a group of r random farms
                    rand_agents = self.farms.random(n=self.p.number_of_friends,replace=True)
                    #Compute the group average belief
                    sum_opinions = sum(rand_agents.O)
                    avg_opinion = sum_opinions/self.p.number_of_friends
                    #Set agents beliefs accordingly to the Majority rule
                    if avg_opinion > 0.5:
                        for i, agent in enumerate(rand_agents):
                            agent.new_O = 1
                    else:
                        pass
                else:
                    raise ValueError("Simulation aborted. Interaction types allowed: Voter or MajorityRule")
            elif isinstance(self.p.interacting_farms, tuple): # l changing num of agents that interact per time step
                #Subset of brown farms
                brown_farms = self.farms.select(self.farms.new_O == 0)
                #select l random brown agents
                l = self.p.interacting_farms[self.t]
                if len(brown_farms) > 0 and l > 0:
                    gen_new_greens = brown_farms.random(n=l)
                    #update farm's opinion
                    for i, agent in enumerate(gen_new_greens):
                        agent.new_O = 1
                else:
                # All farms are green
                    pass
            else: # l fixed num. of agents change opinion at each time step 
                #Subset of brown farms
                brown_farms = self.farms.select(self.farms.new_O == 0)
                if len(brown_farms) < self.p.interacting_farms:
                    pass
                else:
                    #select l random brown agents
                    gen_new_greens = brown_farms.random(n=self.p.interacting_farms)
                    #update farm's opinion
                    for i, agent in enumerate(gen_new_greens):
                        agent.new_O = 1
        else:
            pass
        
        ##------- END INTERACTION PROCESSS
        
        #FARMS update their opinion
        self.farms.update_opinion()
        
        #STATISTICIAN counts farms with opinion 0
        self.statistician.count_farms_opinion()
        
        #FARMS invest
        self.farms.investment()
        #FARMS compute capital
        self.farms.capital()
                
        #STATICIAN Computes Macroeconomic states
        self.statistician.technical_progress()
        self.statistician.aggregate_capital()
        
        #FARMS read information from statistician
        self.farms.read_news()
        
        #FARMS decide labor allocation
        self.farms.labor()
        #FARMS produce
        self.farms.production()


        #STATISTICIAN compute Macroeconomic states
        self.statistician.gross_investment()
        self.statistician.growth_rate()
        self.statistician.gross_domestic_product()
        
        #FARMS consume & compute utility
        self.farms.consumption()        
        self.farms.utility()
        

        #STATISTICIAN computes capital-to-output ratio
        self.statistician.gross_consumption()
        self.statistician.capital_output_ratio()
        
        #STATISTICIAN computes emissions
        self.statistician.count_emissions()
        
       

    def update(self):
    #This function records agents' & Model variables

        # FARMS variables
        self.farms.record("G")
        self.farms.record('B')
        self.farms.record("K")
        #self.farms.record('b')
        #self.farms.record('g')
        #self.farms.record('L')
        self.farms.record("P_G")
        self.farms.record("P_B")
        self.farms.record("P")
        #self.farms.record('I_G')
        #self.farms.record('I_B')
        #self.farms.record('I')
        #self.farms.record('C')
        #self.farms.record('U')
        #self.farms.record('u')
        self.farms.record('new_O')
        
        # STATISTICIAN variables 
        self.statistician.record("Total_G")
        self.statistician.record("Total_B")
        self.statistician.record("Total_K")
        self.statistician.record("real_beta")
        self.statistician.record("real_gamma")
        self.statistician.record("GDP")
        self.statistician.record("Total_P_G")
        self.statistician.record("Total_P_B")
        #self.statistician.record("g_rate")
        #self.statistician.record("Total_I_G")
        #self.statistician.record("Total_I_B")
        #self.statistician.record("Total_I")
        #self.statistician.record("Total_C")
        #self.statistician.record('K_output_rate')
        self.statistician.record('B_farms')
        if self.p.emissions == True:
            self.statistician.record('C_CO2')
        else:
            pass

    def end(self):
        
        #Number of green farms at the end of the simulation
        self.report("new_O", sum(agent.new_O  for agent in self.farms))
        #Final green output share of the economy
        self.report('greenOutputShare', sum(self.farms.P_G)/sum(self.farms.P))

        #Optional: Draw final network
        #Plot network
        if self.p.draw_network == True:
            fig, ax = plt.subplots(figsize=( 215/25.4, 180/(2*25.4),))
            plt.title('Final networs')
            color_dict = {0:'#744700', 1:'#008000'}
            colors = [color_dict[c] for c in self.farms.O]
            nx.draw_circular(self.network.graph, node_color=colors,
                         node_size=50)
        else:
            pass


