import numpy as np

class DefaultFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces, current_aid):

        self._action_space = action_spaces
        self._observation_space = observation_spaces
        self.current_aid = current_aid

    def encode(self, state):
        obs = state['obs']
        controlled_idx = state['controlled_player_index']
        my_agents = obs['agents'][controlled_idx]
        opponent_agents = obs['agents'][1-controlled_idx]

        self.sorted_res = sorted(obs['restaurants'], key=lambda x:x['restaurant_id'])
        self.sorted_cus = sorted(obs['customers'], key=lambda x:x['customer_id'])
        self.current_t = obs['current_step']
        self.current_pos = np.array(my_agents['position'])

        ############# My order that needs to be picked ################
        my_orders_to_pick = []
        for order in my_agents['orders_to_pick']:
            encoded_order = self.order_dispatch(order)
            my_orders_to_pick.append(encoded_order)
        if len(my_orders_to_pick)<20:
            my_orders_to_pick.append(np.zeros(8*(20-len(my_orders_to_pick))))
        my_orders_to_pick = np.concatenate(my_orders_to_pick)
            # my_orders_to_pick.append([0]*8*(20-len(my_orders_to_pick)))     #padding

        ############# My order that has been picked and needs to be delivered ##############33
        my_order_to_deliver = []
        for order in my_agents['order_to_deliver']:
            encoded_order = self.order_dispatch(order)
            my_order_to_deliver.append(encoded_order)
        if len(my_order_to_deliver) < 5:
            my_order_to_deliver.append(np.zeros(8*(5-len(my_order_to_deliver))))
        my_order_to_deliver = np.concatenate(my_order_to_deliver)
            # my_order_to_deliver.append([0]*8*(5-len(my_order_to_deliver)))

        ############# New order that can be accepted #######################################
        new_orders = []
        for order in obs['new_orders']:
            encoded_order = self.order_dispatch(order)
            new_orders.append(encoded_order)
        if len(new_orders) <10:
            new_orders.append([0]*8*(10-len(new_orders)))
        new_orders = np.concatenate(new_orders)

        encoded_obs = np.concatenate(
            [new_orders,
             my_orders_to_pick,
             my_order_to_deliver]           #[80,160,40]
        )

        minimap = self.minimap(obs['roads'], [i['position'] for i in obs['restaurants']],
                               [i['position'] for i in obs['customers']], [my_agents['position'], opponent_agents['position']])

        obs_map = np.concatenate([encoded_obs, minimap.flatten()])      #280+256

        return obs_map, np.ones(40, dtype=np.float32)

    def order_dispatch(self, order):
        order_id = order['order_id']
        customer_id = order['customer_id']
        restaurant_id = order['restaurant_id']
        start_time = order['start_time']
        end_time = order['end_time']

        restaurant_loc = self.sorted_res[restaurant_id]['position']
        customer_loc = self.sorted_cus[customer_id]['position']
        distance2res = self.Manhattan_distance(self.current_pos, restaurant_loc)
        distance2cus = self.Manhattan_distance(self.current_pos, customer_loc)
        distance_res2cus = self.Manhattan_distance(restaurant_loc, customer_loc)

        remaining_t = end_time-self.current_t

        order_info = np.concatenate([restaurant_loc, customer_loc,
                                    [distance2res, distance2cus,
                                     distance_res2cus, remaining_t]])    #dim8
        return order_info

    def minimap(self, road, restaurant, customer, rider):
        # 1: road, 2: restaurant, 3: customer, 4: rider
        map = np.zeros((16,16))
        for ro in road:
            map[ro[0], ro[1]] = 1
        for re in restaurant:
            map[re[0], re[1]] = 2
        for cu in customer:
            map[cu[0], cu[1]] = 3
        for ri in rider:
            map[ri[0],ri[1]] = 4

        return map


    def Manhattan_distance(self, x,y):
        return abs(x[0]-y[0]) + abs(x[1]-y[1])

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
