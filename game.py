import numpy as np
import cv2
import pyzed.sl as sl
from ColorModule import ColorFinder
import math


class Team:
    def __init__(self, name, color, team_color):
        self.name = name
        self.color = color
        self.team_color = team_color
        self.score = 0
        self.players = []  # tuple of 2d x, y coords of each player

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def get_score(self):
        return self.score

    def add_score(self, score):
        self.score += score

    def goal(self):
        self.score += 1

    def get_players(self):
        return self.player

    def add_player(self, player_obj, window_size, real_size):
        self.players.append(generate_2d_obj_position(
            player_obj, window_size, real_size))

    def clear_players(self):
        self.players.clear()

    def update_football_map(self, ground, player_radius):
        for player_coords in self.players:
            cv2.circle(
                ground, player_coords, 20,  self.team_color, -1)
            cv2.circle(
                ground, player_coords, player_radius,  self.team_color, 2)

    def get_last_player_coords(self):
        return self.players[-1]


class Game:
    def __init__(self):
        self.teamA = Team("A",
                          {'hmin': 0, 'smin': 0, 'vmin': 197,
                              'hmax': 179, 'smax': 255, 'vmax': 255},
                          (255, 0, 255))
        self.teamB = Team("B",
                          {'hmin': 0, 'smin': 0, 'vmin': 0,
                              'hmax': 179, 'smax': 255, 'vmax': 255},
                          (0, 255, 0))

        self.color_finder = ColorFinder(False)

        # {pos: (x, y), with: ("team": team, "player": player_obj), "kickzone": "color"}
        self.ball_data = {"pos": (720, 720), "team": None, "kickzone": None}

        # SIZE OF THE VIRTUAL GROUND. MAKE SURE ALWAYS SQUARE
        self.window_size = (720, 720)

        # Real size (as a square) of the football ground in meters
        self.real_size = (8, 8)

        self.virtual_ground_radius = self.window_size[0] // 2
        self.goal_radius = 40
        self.player_radius = 50

        self.center = (self.virtual_ground_radius, self.virtual_ground_radius)
        self.red_radius = self.virtual_ground_radius
        self.yellow_radius = self.virtual_ground_radius * 2 // 3
        self.green_radius = self.virtual_ground_radius // 3
        self.blue_radius = self.virtual_ground_radius // 4

    def generate_football_map(self, image, objects, is_tracking_on):
        # Draws the football ground:
        ground = np.zeros(
            shape=(self.window_size[0], self.window_size[1], 3), dtype=np.uint8)

        cv2.circle(
            ground, self.center, self.red_radius, (0, 0, 255), -1)
        cv2.circle(ground, self.center, self.yellow_radius, (0, 255, 255), -1)
        cv2.circle(ground, self.center, self.green_radius, (0, 255, 0), -1)
        cv2.circle(ground, self.center, self.blue_radius, (255, 0, 50), -1)
        cv2.rectangle(ground, (self.center[0]-2, self.center[1]-10),
                      (self.center[0]+2, self.center[1]+10), (255, 255, 255), -1)

        cv2.circle(ground, self.center, self.goal_radius, (255, 0, 255), 2)

        # Draws the players on the ground:
        self.teamA.clear_players()
        self.teamB.clear_players()

        for obj in objects.object_list:
            # ID DONT KNOW WHY, BUT REMOVING THIS BREAKS THE CODE!!?? JUST DON'T REMOVE IT!!
            if render_object(obj, is_tracking_on):
                try:
                    if obj.label == sl.OBJECT_CLASS.PERSON:
                        team = self.determine_team(image, obj)
                        team.add_player(
                            obj, self.window_size, self.real_size)

                        # Determine if player has ball and update the ball's data
                        self.update_ball_data(team)
                    else:
                        self.ball_data["pos"] = generate_2d_obj_position(
                            obj, self.window_size, self.real_size)

                except Exception as e:
                    print(repr(e))
                    print("LINE 115")

        try:
            # check for goal
            if euclidean_distance(self.ball_data["pos"], self.center) <= self.goal_radius:
                if self.ball_data["team"]:
                    if self.ball_data["kickzone"] == "red":
                        self.ball_data["team"].add_score(3)
                        print("GOAL BY TEAM:",
                              self.ball_data["team"].get_name(), "for 3 points!")
                    elif self.ball_data["kickzone"] == "yellow":
                        self.ball_data["team"].add_score(2)
                        print("GOAL BY TEAM:",
                              self.ball_data["team"].get_name(), "for 2 points!")
                    elif self.ball_data["kickzone"] == "green":
                        self.ball_data["team"].add_score(1)
                        print("GOAL BY TEAM:",
                              self.ball_data["team"].get_name(), "for 1 points!")
                    else:
                        print("GOAL FROM PENALTY ZONE BY TEAM:",
                              self.ball_data["team"].get_name())
                    self.ball_data["team"] = None

        except Exception as e:
            print(repr(e))
            print("LINE 136")
            # print(repr(e))

        # Draws each teams players and the ball on the ground
        self.teamA.update_football_map(ground, self.player_radius)
        self.teamB.update_football_map(ground, self.player_radius)
        cv2.circle(ground, self.ball_data["pos"], 10, (255, 255, 255), -1)

        return ground

    def generate_scoreboard(self):
        window = np.zeros(shape=(self.virtual_ground_radius*2,
                          self.virtual_ground_radius*2, 3), dtype=np.uint8)
        cv2.putText(window, "Scoreboard", (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(window, "Ball With: Team A" if self.ball_data["team"] == self.teamA else "Ball With: Team B", (
            10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(window, "Team A: " + str(self.teamA.get_score()),
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(window, "Team B: " + str(self.teamB.get_score()),
                    (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.putText(window, self.ball_data["kickzone"],
                    (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return window

    def determine_team(self, image, obj):
        top_left_corner = self.cvt(obj.bounding_box_2d[0])
        top_right_corner = self.cvt(obj.bounding_box_2d[1])
        bottom_right_corner = self.cvt(obj.bounding_box_2d[2])
        bottom_left_corner = self.cvt(obj.bounding_box_2d[3])

        roi_width = int(top_right_corner[0] - top_left_corner[0])
        roi_height = int(
            (bottom_left_corner[1] - top_left_corner[1]) * (3 / 5))
        roi = image[int(top_left_corner[1]):int(
            top_left_corner[1] + roi_height), int(top_left_corner[0]):int(top_left_corner[0] + roi_width)]

        _, maskA = self.color_finder.update(roi, self.teamA.get_color())
        _, maskB = self.color_finder.update(roi, self.teamB.get_color())
        # cv2.imshow("maskA", maskA)
        # cv2.imshow("maskB", maskB)

        if (maskA.sum() > maskB.sum()):
            return self.teamA
        else:
            return self.teamB

    def update_ball_data(self, team):
        player_2d_coords = team.get_last_player_coords()
        if euclidean_distance(player_2d_coords, self.ball_data["pos"]) <= self.player_radius:
            self.ball_data["team"] = team
            center_distance = euclidean_distance(self.center, player_2d_coords)
            if center_distance <= self.red_radius and center_distance > self.yellow_radius:
                self.ball_data["kickzone"] = "red"
            elif center_distance <= self.yellow_radius and center_distance > self.green_radius:
                self.ball_data["kickzone"] = "yellow"
            elif center_distance <= self.green_radius and center_distance > self.blue_radius:
                self.ball_data["kickzone"] = "green"
            else:
                self.ball_data["kickzone"] = "blue"

    def cvt(self, pt):
        out = [pt[0], pt[1]]
        return out


def generate_2d_obj_position(obj, window_size, real_size):
    window_width, window_height = window_size
    real_width, real_depth = real_size

    x_pos = np.interp(
        obj.position[0], (-real_width/2, real_width/2), (0, window_width))
    y_pos = np.interp(abs(obj.position[2]), (
        0, real_depth), (window_height, 0))

    return int(x_pos), int(y_pos)


def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
