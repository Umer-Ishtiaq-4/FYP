import re
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import random
# from transformers import pipeline

class InputProcessor:
    def __init__(self):
        self.Room_Class = ['livingroom', 'bedroom', 'corridor', 'kitchen', 'washroom', 'study', 'closet', 'storage', 'balcony']
        self.graph = nx.Graph()

    def parse_input(self, input_text):
        sentences = re.split('[.!?]+', input_text)
        for sentence in sentences:
            for room_type in self.Room_Class:
                if room_type in sentence.lower():
                    room_name = self.extract_room_name(sentence, room_type)
                    if room_name is not None:
                        self.graph.add_node(room_name, room_type=room_type)

            if "connected" in sentence or "next to" in sentence or "adjacent to" in sentence or "and" in sentence:
                connections = self.extract_connections(sentence)
                for connection in connections:
                    self.graph.add_edge(*connection)
            

    def extract_room_name(self, sentence, room_type):
        match = re.search(f"({room_type}\d+)", sentence, re.IGNORECASE)
        if match:
            return match.group(0)
        else:
            return None

    def extract_connections(self, sentence):
        room_names = re.findall(r"([a-z]+[\d]+)", sentence, re.IGNORECASE)
        connections = []
        for i in range(len(room_names) - 1):
            connections.append((room_names[i], room_names[i + 1]))
        return connections

    def generate_output(self):
        rooms = []
        links = []
        rooms.append(list(self.graph.nodes))
        # 2d -> 1d
        rooms = [item for sublist in rooms for item in sublist]

        for a, b in self.graph.edges:
            links.append([a,b])
        # rooms_output = "[Rooms]\n" + "\n".join(self.graph.nodes) + "\n"
        # links_output = "[Links]\n" + "\n".join([f"{a}, {b}" for a, b in self.graph.edges]) + "\n"
        rooms_w_link_dict = {
            'rooms': rooms,
            'links': links
        }
        return rooms_w_link_dict



def generate_output(self):
    rooms = []
    links = []
    pos = {}
    for node in self.graph.nodes:
        room_type = self.graph.nodes[node]['room_type']
        x, y = node[room_type], node[room_type + 1]
        pos[node] = (int(x), int(y))
        rooms.append({'name': node, 'x': int(x), 'y': int(y)})
    for a, b in self.graph.edges:
        links.append({'source': a, 'target': b})
    output = {'rooms': rooms, 'links': links}
    return output


def post_processing_on_text(output):
    position_classes = ['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE']
    room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen', 'washroom', 'study', 'closet', 'storage', 'balcony']
    # room_sizes = {
    #     'livingroom': random.uniform(25, 45),
    #     'bedroom': random.uniform(6, 20),
    #     'corridor': random.uniform(2, 5),
    #     'kitchen': random.uniform(6, 12),
    #     'washroom': random.uniform(4, 10),
    #     'study': random.uniform(4, 16),
    #     'storage': random.uniform(2, 8),
    #     'balcony': random.uniform(3.5, 8),
    #     'closet': random.uniform(3, 10)
    # }
    
    positions_visited = {
        'NW': -1,
        'N' : -1,
        'NE' : -1,
        'W' : -1,
        'C' : -1,
        'E' : -1,
        'SW' : -1,
        'S' : -1,
        'SE': -1
    }
    size_positions = {}

    # print(positions_visited[position_classes[random.randint(0, 8)]])
    for room in output['rooms']:
        count = 0 
        if room[:-1] == 'livingroom':
            size = round(random.uniform(25, 45), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position] 

        elif room[:-1] == 'bedroom':
            size = round(random.uniform(6, 20), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'corridor':
            size = round(random.uniform(2, 5), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'kitchen':
            size = round(random.uniform(6, 12), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'washroom':
            size = round(random.uniform(4, 10), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'study':
            size = round(random.uniform(4, 16), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'closet':
            size = round(random.uniform(3, 10), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'storage':
            size = round(random.uniform(2, 8), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]

        elif room[:-1] == 'balcony':
            size = round(random.uniform(3.5, 8), 3)
            position = position_classes[random.randint(0, 8)]
            positions_visited[position] = positions_visited[position] + 1
            while(positions_visited[position] > 1 and count < 7):
                position = position_classes[random.randint(0, 8)]
                count = count + 1
            size_positions[room] = [size, position]
    
    output['sizes'] = size_positions
    # pprint.pprint(output)


def process_input(input_text, draw=False):
    layout_generator = InputProcessor()
    layout_generator.parse_input(input_text)
    processed_input = layout_generator.generate_output()
    post_processing_on_text(processed_input)


    # pprint.pprint(output)
    # print(type(output))
    if draw:
        G = nx.Graph()
        for room_name in processed_input['rooms']:
            G.add_node(room_name)

        for link in processed_input['links']:
            G.add_edge(*link)

        pos = nx.spring_layout(G)  # you can change the layout algorithm here

        nx.draw(G, pos, with_labels=True, node_size=500)

        plt.show()
    return processed_input


def is_single_precision(num):
    # Convert the number to a string
    num_str = str(num)

    # Check if the string contains a decimal point
    if '.' not in num_str:
        return False

    # Split the string at the decimal point and check the length of the second part
    _, decimal_part = num_str.split('.')
    return len(decimal_part) == 1


def check_sizes(exxtracted_info):
    sizes_dict_passed = exxtracted_info['sizes']
    for room in sizes_dict_passed:
        # print(room)
        if exxtracted_info['sizes'][room][0] == 0 or exxtracted_info['sizes'][room][0] == None or is_single_precision(exxtracted_info['sizes'][room][0]):
            if room[:-1] == 'livingroom':
                sizes_dict_passed[room][0] =  round(random.uniform(25, 45),3)
            elif room[:-1] == 'bedroom':
                sizes_dict_passed[room][0] =  round(random.uniform(6, 20),3)
            elif room[:-1] == 'corridor':
                sizes_dict_passed[room][0] =  round(random.uniform(2, 5),3)
            elif room[:-1] == 'kitchen':
                sizes_dict_passed[room][0] =  round(random.uniform(6, 12),3)
            elif room[:-1] == 'washroom':
                sizes_dict_passed[room][0] =  round(random.uniform(4, 10),3)
            elif room[:-1] == 'study':
                sizes_dict_passed[room][0] =  round(random.uniform(4, 16),3)
            elif room[:-1] == 'storage':
                sizes_dict_passed[room][0] =  round(random.uniform(2, 8),3)
            elif room[:-1] == 'balcony':
                sizes_dict_passed[room][0] =  round(random.uniform(3.5, 8),3)
            elif room[:-1] == 'closet':
                sizes_dict_passed[room][0] =  round(random.uniform(3, 10),3)

    # print(exxtracted_info)
    return exxtracted_info

# input_text = """
#     The building contains three bedrooms, one washroom, two balconys, one livingroom, and one kitchen. livingroom1 is adjacent to bedroom1, livingroom1 is adjacent to bedroom2, livingroom1 is adjacent to bedroom3, livingroom1 is adjacent to kitchen1, livingroom1 is adjacent to balcony1, livingroom1 is adjacent to washroom1. bedroom2 and bedroom1 are connected. bedroom2 is next to washroom1. washroom1 and bedroom3 are connected. kitchen1 is next to balcony2. 
#     """
# process_input(input_text)