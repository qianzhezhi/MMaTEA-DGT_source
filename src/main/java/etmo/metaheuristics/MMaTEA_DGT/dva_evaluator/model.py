from datetime import datetime
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from copy import deepcopy
from os import path

PATH_DATA = 'data/input'

def read_data_table(filename):
    if filename.endswith('.xlsx'):
        return pd.read_excel(path.join(PATH_DATA, filename))
    elif filename.endswith('.csv'):
        return pd.read_csv(path.join(PATH_DATA, filename))
    else:
        raise FileNotFoundError(f'The file type of {filename} is not supported.')

def get_pop_data(area_name, city_name):
    pop, tier1 = None, None

    if area_name.lower() == 'default':
        if city_name.lower() == 'default':
            df = read_data_table('population_china.xlsx')
            pop = df['Total'].to_numpy()
            tier1 = df["Tier12"].to_numpy()
        else:
            df = read_data_table('population_china_cities2.csv')
            pop = df[city_name.capitalize()].to_numpy()
            tier1_rate = df['tier1_rate'].to_numpy()
            tier1 = pop * tier1_rate
    else:
        df = read_data_table('poptotal_177.csv')
        row = df[df['iso3c'] == area_name.upper()]
        pop = row.to_numpy()[0][2:-1]
        tier1 = np.zeros_like(pop)

    return pop, tier1

def get_cm_data(area_name):
    cm = None
    df = read_data_table('suscept_age.csv')
    s = df['sus'].to_numpy()

    if area_name.lower() == 'default':
        df = read_data_table('cm_china.xlsx')
        cm = df.to_numpy()[:, 1:]
        cm = s * cm
    else:
        df = read_data_table('cm_177.csv')
        cm = df[[f'{area_name.upper()}.{i}' for i in range(1, 17)]].to_numpy()
        group_size = 16
        s[group_size-1] = np.mean(s[group_size-1:])
        s = s[:group_size]
        s = s / np.max(s)
        cm = s * cm

    return cm

def get_vaccine_coverage_data(area, type):
    df = read_data_table(f'vaccine_coverage_{type}.xlsx')
    v1, v2, v3 = df['V12'].to_numpy(), df['V2'].to_numpy(), df['V3'].to_numpy()
    v1 = v1 - v2
    if not area == 'default':
        v1[-2] += v1[-1]
        v1 = v1[:-1]
        v2[-2] += v2[-1]
        v2 = v2[:-1]
        v3[-2] += v3[-1]
        v3 = v3[:-1]
    return v1, v2, v3

def get_risk_data(area_name, variant):
    df = read_data_table(f'disease_burden_{variant}.xlsx')
    risk_data = {
        'infec': df['infec'].to_numpy(),
        'symp': df['symp'].to_numpy(),
        'hosp': df['hosp'].to_numpy(),
        'icu': df['icu'].to_numpy(),
        'death': df['death'].to_numpy(),
    }
    if area_name.lower() != 'default':
        for risk_type in risk_data:
            risk_data[risk_type][-2] = np.mean(risk_data[risk_type][-2:])
            risk_data[risk_type] = risk_data[risk_type][:-1]
    return risk_data


class SEIR3_faculty():

    def __init__(self, area='default', city='default', variant='delta', vc=None, max_day=400):
        self.area = area
        self.city = city
        self.variant = variant
        self.vc = vc
        self.max_day = max_day

    def make_problem(self):
        model = None
        if self.vc:
            model = SEIR3(self.area, self.city, self.variant, vc=self.vc, max_day=self.max_day)
        else:
            model = SEIR3(self.area, self.city, self.variant, max_day=self.max_day)

        def problem(x, risk_type):
            model.run(x)
            return model.get_fitness(risk_type)
        return problem

    def make_problem_of_MO(self):
        model = None
        if self.vc:
            model = SEIR3(self.area, self.city, self.variant, vc=self.vc)
        else:
            model = SEIR3(self.area, self.city, self.variant)

        def problem(x):
            # x = np.array(x).astype('float32')
            model.run(x)
            return [model.get_fitness('infec'), model.get_fitness('death')]
        return problem

    def make_model(self,area='default'):
        return SEIR3(self.area, self.city, self.variant, vc=self.vc)



class SEIR3:

    def __init__(self, area_name="default", city_name='default',  variant='delta', v2_decline=6, v3_decline=3, **other):
        '''
        Store parameters and read data from file
        '''
        # ===== Scenarios START ======
        # R0 = 7.2, 6.0

        R0 = 6.0
        if variant == 'omicron':
            R0 = 7.2

        if 'R0' in other:
            print(f'custom R0 detected: {other["R0"]}')
            R0 = other['R0']
        
        Rt = 1.5
        R_time = 21
        self.Rt = np.interp(range(R_time), [0, R_time-1], [R0, Rt]).tolist() + [1.5] * (400 - 21)

        # rollouts = Delta: 0.0017, Omicron: 0.0028
        # ROLLOUTS = 0.0017
        volumn = 4000000
        rollouts = volumn / 1439323774
        # if variant == 'omicron':
        #     ROLLOUTS = 0.0028

        # delay = 0, 30, 60, 90, 120
        DELAY = 0
        # willing = 1, 0.83, 0.61
        VACCINE_WILLING = 1
        # ===== Scenarios END ======

        self.max_day = 400
        if 'max_day' in other:
            self.max_day = other['max_day']
        self.gamma = 1 / 4.6

        # incubation
        self.lmd = 1 / 3.0

        self.omega_12 = 1 / (21 + 14)
        self.omega_3 = 1 / 14

        self.R = R0

        self.N, self.init_tier1 = get_pop_data(area_name, city_name)
        self.group_size = len(self.N)

        # Delta
        self.VE1 = 0.138
        self.VE2 = 0.51
        self.VE2W = 0.331
        self.VE3 = 0.79
        self.VE3W = 0.641

        if variant == 'omicron':
            self.VE1 = 0.048
            self.VE2 = 0.091
            self.VE2W = 0.059
            self.VE3 = 0.567
            self.VE3W = 0.459

        self.V2_waning_time = v2_decline
        self.V3_waning_time = v3_decline

        self.compartments = \
            ['S', 'V1'] + \
            [f'V2{i}' for i in range(self.V2_waning_time + 1)] + \
            [f'V3{i}' for i in range(self.V3_waning_time + 1)] + \
            ['I', 'R']

        # init_I = [1] * self.group_size
        init_I = [0] * self.group_size
        init_I[5] = 1
        p_V1, p_V2, p_V3 = get_vaccine_coverage_data(area_name, variant)

        if variant == 'omicron':
            if not 'vc' in other:
                # print("Error: no vc parameter, setting default to 0.3...")
                other['vc'] = 0.3
            vc = other['vc']
            p_V1 = np.ones_like(self.N) * vc * 0.1
            p_V2 = np.ones_like(self.N) * vc * 0.9

        init_V1 = np.floor(self.N * p_V1).astype('int64')

        init_V2 = [np.zeros(self.group_size) for _ in range(self.V2_waning_time + 1)]
        if np.sum(p_V2) > 0:
            vol_V2 = np.floor(self.N * p_V2).astype('int64')
            # vol_monthly_max = ROLLOUTS * 30 * self.N

            l = 4
            if variant == 'omicron':
                l = len(init_V2)

            for i in range(l):

                init_V2[i] = vol_V2 / l 

        init_V3 = [np.zeros(self.group_size) for _ in range(self.V3_waning_time + 1)]

        self.init_tier1 = np.floor(self.init_tier1 * (np.ones_like(p_V1) - p_V1 + p_V2 + p_V3)).astype('int64')

        if area_name == 'default' and city_name == 'default':
            self.vaccine_daily_volumn_max = volumn
        else:
            self.vaccine_daily_volumn_max = np.floor(np.sum(self.N) * rollouts).astype('int64') 

        array_S = self.N - init_I - (self.N * (p_V1 + p_V2 + p_V3))
        array_V1 = init_V1
        array_V2 = np.concatenate(init_V2)
        array_V3 = np.concatenate(init_V3)
        array_I = init_I
        array_R = np.zeros_like(init_I)

        self.init_state = np.concatenate([
            array_S, array_V1, array_V2, array_V3, array_I, array_R
        ])
        assert len(self.init_state) == len(self.compartments) * self.group_size

        self.risk_data = get_risk_data(area_name, variant)
        self.C = get_cm_data(area_name)

        self.betas = [self.get_beta(rt) for rt in self.Rt]

        e_decline = np.array([0.75] * 3 + [1] * 9 + [0.75] * (self.group_size - 3 - 9))
        VE2s = np.interp(range(self.V2_waning_time + 1), [0, 6], [self.VE2, self.VE2W])
        VE3s = np.interp(range(self.V3_waning_time + 1), [0, 6], [self.VE3, self.VE3W])

        self.e = np.zeros((3 + self.V2_waning_time + self.V3_waning_time, self.group_size))
        self.e[0] = self.VE1 * e_decline
        for i, e in enumerate(VE2s):
            self.e[i+1] = e * e_decline
        for i, e in enumerate(VE3s):
            self.e[i+2+self.V2_waning_time] = e * e_decline

        # self.vaccinated_willing = np.ones(self.group_size) * VACCINE_WILLING
        self.individual_reject_vaccine = ((self.init_state[:self.group_size] - self.init_tier1) * (1 - VACCINE_WILLING)).astype("int64")

        self.model = self.get_seir_model()
        # reset state
        self.states = []
        self.Vs = []
        self.tier1s = []
        self.reset_state()

    def get_beta(self, Rt):
        return np.round(Rt * self.gamma / np.abs(np.max(np.linalg.eig(self.C.astype('float'))[0])), 7)

    def reset_state(self):
        self.states.clear()
        self.Vs.clear()
        self.tier1s.clear()

    def get_state(self):
        return deepcopy(self.states)

    def get_state_of(self, day):
        return self.states[day].copy()

    def run(self, input):
        if isinstance(input, list):
            input = np.array(input)
        # initial state 
        if input.ndim == 1:
            x = input.reshape(-1, self.group_size)
        else:
            x = input

        if x.shape[0] < 400:
            days_x = 400 // x.shape[0]
            tmp = x.repeat(days_x, axis=0)
            left = 400 - tmp.shape[0]
            tmp = np.vstack([tmp, x[-1].reshape(1, -1).repeat(left, axis=0)])
            x = tmp

        self.reset_state()
        state = self.init_state.copy()
        tier1 = self.init_tier1.copy()

        self.states.append(state)
        self.tier1s.append(tier1)

        for i, daily_x in enumerate(x):

            state, tier1, v = self.step(daily_x, state, tier1, i)

            self.states.append(state)
            self.tier1s.append(tier1)
            self.Vs.append(v)
        
    def step(self, x, state_input, tier1_input, i):
        assert len(x) == self.group_size
        assert len(state_input) == self.group_size * len(self.compartments)

        if np.isnan(x).any():
            print(f"Warning: illegal variables: {x}")

        state = state_input.copy()
        tier1 = tier1_input.copy()
        
        vaccine_daily_volumn = self.vaccine_daily_volumn_max - np.sum(state[1*self.group_size:2*self.group_size] * self.omega_12).astype('int64')
        # # DEBUG
        # print(f'{self.vaccine_daily_volumn} / {self.vaccine_daily_volumn_max}')

        if np.sum(state[:self.group_size]) <= vaccine_daily_volumn:
            v = state[:self.group_size].copy()
        else:
            v = self.decode(x, state, tier1, vaccine_daily_volumn)
        
        tier1 = (tier1 - v).clip(min=0)
        
        gz = self.group_size
        # S
        state[0*gz:1*gz] = state[0*gz:1*gz] - v
        # V
        state[1*gz:2*gz] = state[1*gz:2*gz] + (1 - self.omega_12) * v

        daily_sol = solve_ivp(self.model, [0, 1], state, t_eval=[1], rtol=1e-6, atol=1e-6, method="LSODA",
                              args=(self.betas[i], self.gamma, self.omega_12, self.omega_3, 1/30, self.e, self.C, np.zeros_like(v)))

        return daily_sol.y.T[0], tier1, v

    def get_fitness(self, risk_type):
        return self.calculate_fitness(self.states, risk_type)

    def calculate_fitness(self, states, risk_type):
        risk_type = risk_type.lower()
        if not risk_type in self.risk_data.keys():
            print(f'Error: {risk_type} is not in {self.risk_data.keys()}')
        if len(states) < 2:
            print(f'Warining: the states length is less than 2.')
        fitness = 0
        days = len(states)
        for day in range(days-1):
            # print(f"{day}: {fitness}")
            fitness += np.sum(
                self.risk_data[risk_type] *
                (states[day+1][-2*self.group_size:-1*self.group_size] -
                (1 - self.gamma) * states[day][-2*self.group_size:-1*self.group_size]))

        return fitness

    def calculate_myopic_objective(self, x, state, tier1, risk_type, repeat_time=1):
        # 不需要为tier1加惩罚，因为step的运行已满足约束 
        myopic_objective = 0
        new_state = state
        new_tier = tier1
        for i in range(repeat_time):
            new_state, new_tier, _ = self.step(x, new_state, new_tier, i)
            new_I = new_state[-2*self.group_size:-1*self.group_size]
            myopic_objective += np.sum(self.e[1] * self.risk_data[risk_type] * (self.C @ (new_I / np.sum(new_state))))

        return myopic_objective

    def save_states(self, states, file_name):
        head = []
        for C in self.compartments:
            head.extend([f'{C}_{i+1}' for i in range(self.group_size)])
        df = pd.DataFrame(data=np.vstack(states), columns=head)
        df.to_csv(
            f"data/output/output_{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    def save_allocation(self, x, file_name):
        head = [f"allocation_{i+1}"for i in range(self.group_size)]
        df = pd.DataFrame(data=x.reshape(self.max_day, -1), columns=head)
        df.to_csv(
            f"data/output/alloca_{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    
    def get_seir_model(self):
        def SVIR(t, z, beta, gamma, omega12, omega3, m, e, C, v):
            comps = z.reshape(len(self.compartments), -1)
            # calculate a duplicating factor
            N = np.sum(comps, 0)
            B = beta * C @ (comps[-2] / N)

            d = np.zeros_like(comps)

            # S
            d[0] = -v - comps[0] * B
            # V1
            d[1] = v - omega12 * comps[1] - (1 - e[0]) * comps[1] * B
            # V2_0 -> V2_k1
            d[2] = omega12 * comps[1] - m * comps[2] - (1 - e[1]) * comps[2] * B
            i = 3
            while i < 2 + self.V2_waning_time:
                d[i] = m * comps[i-1] - m * comps[i] - (1 - e[i-1]) * comps[i] * B
                i += 1
            d[i] = m * comps[i-1] - omega3 * comps[i] - (1 - e[i-1]) * comps[i] * B
            # V3_0 -> V3_k2
            i += 1
            d[i] = omega3 * comps[i-1] - m * comps[i] - (1 - e[i-1]) * comps[i] * B
            i += 1
            while i < 3 + self.V2_waning_time + self.V3_waning_time:
                d[i] = m * comps[i-1] - m * comps[i] - (1 - e[i-1]) * comps[i] * B
                i += 1
            d[i] = m * comps[i-1] - (1 - e[i-1]) * comps[i] * B
            # # E
            # d[-3] = comps[0] + np.sum(comps[i+1] * (1 - e[i]) for i in range(len(e)))
            # d[-3] = d[-3] * B - comps[-3] * lmd
            # # I
            # d[-2] = comps[-3] * lmd - comps[-2] * gamma
            # I
            d[-2] = comps[0] + np.sum(comps[i+1] * (1 - e[i]) for i in range(len(e)))
            d[-2] = d[-2] * B - comps[-2] * gamma
            # R
            d[-1] = gamma * comps[-2] 

            return d.reshape(-1)
        return SVIR

    def get_risk_data(self, risk_type):
        return self.risk_data[risk_type]

    def problem_maker(self, risk_type):
        def problem(x):
            self.run(x)
            return self.calculate_fitness(self.states, risk_type)
        return problem
    
    def decode(self, x, state, tier1, vaccine_daily_volumn):
        assert len(x) == self.group_size
        if np.sum(x) == 0:
            return x

        if np.sum(tier1) >= vaccine_daily_volumn:
            # T1人数超过每日配额
            # Method 1: 按对应位置的解的比例来分配T1疫苗

            # T1人群已全部接种疫苗，不需要参与分配
            invalid = tier1 <= 0
            valid = np.logical_not(invalid)
            t1_count = np.count_nonzero(valid)
            x_masked = x.copy()
            x_masked[invalid] = 0
            x_sum = np.sum(x_masked)
            if x_sum == 0:
                x_normalized = x_masked
                x_normalized[valid] = 1.0 / t1_count
            else:
                x_normalized = np.array([xx / x_sum for xx in x_masked])
            v = x_normalized * vaccine_daily_volumn
            # 计算疫苗预分配数量与实际人数的差别，进行调整
            left = v - tier1
            overflow = left > 0
            cnt = 0
            while (np.any(overflow)):
                # print(f'T1 vaccine allocation overflow') 
                unfilled = np.logical_not(overflow)
                v[overflow] = tier1[overflow]
                v[unfilled] += ((left[unfilled] / np.sum(left[unfilled])
                                * np.sum(left[overflow])))
                left = v - tier1
                overflow = left > 0
                cnt += 1
                if cnt > 100:
                    print(f'Warning: T1 allocation overflow more than 100 times.')
                    break
            # Method 2: 按照人数比例均匀分配
            # x_sum = np.sum(tier1)
            # v = np.array([xx / x_sum for xx in tier1]) * self.vaccine_daily_volumn
        else:
            # 给T1预先分配疫苗数
            pre_alloc = tier1.copy()
            # 给T2的可分配疫苗数
            vac = vaccine_daily_volumn - np.sum(pre_alloc)
            x_normalized = x.copy()
            # # # 原T1分配不参与二次分配
            # x_normalized[tier1>0] = 0
            # 按照解对T2分配疫苗
            x_sum = np.sum(x_normalized)
            if x_sum < 1e-10:
                return np.zeros_like(x)
            x_normalized = np.array([xx / x_sum for xx in x_normalized])
            v = x_normalized * vac

            state_S = (state[:self.group_size] - self.individual_reject_vaccine).clip(min=0)

            left = v - state_S
            overflow = left > 0
            cnt = 0
            while (np.any(overflow)):
                if np.all(left >= 0):
                    v = state_S[:]
                    break
                # print(f'T2 vaccine allocation overflow')
                unfilled = left < 0
                v[overflow] = state_S[overflow]
                v[unfilled] = v[unfilled] + (left[unfilled] / np.sum(left[unfilled]) * np.sum(left[overflow]))
                left = v - state_S
                overflow = left > 0
                cnt += 1
                if cnt > 100:
                    print(f'Warning: T2 allocation overflow more than 100 times.')
                    break
            v = v + pre_alloc
        
        try:
            assert np.sum(v) <= vaccine_daily_volumn + 1e-6
        except AssertionError as e:
            print(f'{np.sum(x)=}')
            print(f'{np.sum(v)=}')
        
        return v

    def demo(self, type='none'):
        # initial state

        self.reset_state()

        state = self.init_state.copy()
        tier1 = self.init_tier1.copy()

        self.states.append(state)
        self.tier1s.append(tier1)

        # print(f'daily v: {self.vaccine_daily_volumn_max}')
        # print(f"unvaccinated: {np.sum(state[0:17])}")
        # print(f"vaccinated 1: {np.sum(state[1*17:2*17])}")
        # print(f"vaccinated 2: {np.sum(state[2*17:9*17])}")
        # print(f"vaccinated 3: {np.sum(state[9*17:16*17])}")
        # print(f'evaluate vaccination time: {np.sum(state[0:17]) / self.vaccine_daily_volumn_max}')

        contact_eigenvalues = None
        if type == 'contact-based':
            contact_eigenvalues = np.abs(np.linalg.eig(self.C.astype('float'))[0])
            # contact_eigenvalues = contact_eigenvalues - np.min(contact_eigenvalues)

        daily_x = None 
        if type == 'none':
            daily_x = np.zeros(self.group_size)
        elif type == 'avg':
            # if np.sum(state[:self.group_num]) == 0:
            #     print('pause')
            daily_x = state[:self.group_size] / (np.sum(state[:self.group_size]) + 1e-13)
        elif type == '20-':
            daily_x = np.array(((self.init_state[:4] / (np.sum(self.init_state[:4]) + 1e-13))).tolist() + [0] * (self.group_size - 4))
        elif type == '20-49':
            daily_x = np.array([0] * 4 + ((self.init_state[4:10] / (np.sum(self.init_state[4:10]) + 1e-13))).tolist() + [0] * (self.group_size - 10))
        elif type == '60+':
            daily_x = np.array([0] * (self.group_size - 4) + ((self.init_state[:self.group_size][-4:] / (np.sum(self.init_state[:self.group_size][-4:]) + 1e-13))).tolist())
        elif type == 'contact-based':
            daily_x = contact_eigenvalues / np.sum(contact_eigenvalues)
        else:
            print(f"Unknown type: {type}")
            exit(1)
        daily_x = daily_x.astype('float')

        for i in range(self.max_day):

            state, tier1, v = self.step(daily_x, state, tier1, i)

            self.states.append(state)
            self.tier1s.append(tier1)
            self.Vs.append(v)
