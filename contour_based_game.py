import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

#Helper function
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img

def get_contours(bg_img, image, threshold=10):
    
    # abs diff betn img and bg
    diff = cv2.absdiff(bg_img.astype("uint8"), image)    
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt

def show(img, figsize=(10, 10)):
    figure = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

# Preparing the ball
class Ball:
    def __init__(self, window_size=(525, 700, 3), rad=30, position=(255, 300), color=[10, 210, 100], extreme_pos=None):
        """
            window_size: Default viewport
            rad: radius of ball
            position: center of circle
            color: color of the circle
            extreme_pos: to prevent flow from boundary
                    
        """
        
        self.window_size = window_size
        self.rad = rad
        self.position = position
        self.color = color
        self.direction = -90/180 * np.pi # 90 degree
        self.speed = 8 # px per frame on that dirn
        self.hit = False
        self.extreme_pos = extreme_pos
        self.ball_stat = "pause"
        if extreme_pos is None:
            self.extreme_pos = ((15, 15), (window_size[0]-15, window_size[1]-15))
        
    def get_ball(self):
        self.set_ball_position()
        ball_img = np.zeros(self.window_size).astype(np.uint8)
        cv2.circle(ball_img, self.position, self.rad, self.color, -3)
        
        return ball_img
    
    def set_direction(self, p1=None, p2=None, direction=None, kind="h"):
        """
            Used to set direction.
            p1: point position of ball
            p2: position of striking object
            kind: type of strike, horizontal/vertical
            direction: angle with striking body
            
            direction: is none whil usinge p1, p2
        """
        
        if direction is not None:
            if kind == "v":
                self.direction = np.pi-self.direction
            else:
                self.direction = 2*np.pi-self.direction
                
            
        if p1 is not None and p2 is not None:
            if kind == "v":
                self.direction = np.pi-self.direction
            else:
                self.direction = 2*np.pi-self.direction
            
            if p1[1] <= p2[1]:
                self.direction /= 2
            if p1[1] > p2[1]:
                self.direction += self.direction/2
            
        
    def set_ball_position(self):
        extreme_pos = self.extreme_pos
        npos = [int(self.position[1] + np.sin(self.direction)*self.speed), 
                 int(self.position[0] + np.cos(self.direction)*self.speed)]
        npos1 = npos.copy()
        strike_pos=None
        strike_kind = "h"
        if npos[0] <= extreme_pos[0][0]:
            npos[0] = extreme_pos[0][0]
            strike_pos = (extreme_pos[0][0], npos[1])
            strike_kind = "h"
        elif npos[0] >= extreme_pos[1][0]:
            npos[0] = extreme_pos[1][0]
            strike_pos = (extreme_pos[1][0], npos[1])
            strike_kind = "h"
            self.ball_stat = "over"
            #print(self.ball_stat)
            
        if npos[1] <= extreme_pos[0][1]:
            npos[1] = extreme_pos[0][1]
            strike_pos = (npos[0], extreme_pos[0][1])
            strike_kind = "v"
            
        elif npos[1] >= extreme_pos[1][1]:
            npos[1] = extreme_pos[1][1]
            strike_pos = (npos[0], extreme_pos[1][1])
            strike_kind = "v"
            
        if npos != npos1:
            self.set_direction(p1=strike_pos, p2=self.position, direction=self.direction, kind=strike_kind)
        self.position = (npos[1], npos[0])
    
ball = Ball()
show(ball.get_ball())
ball.position = (100, 100)
ball.set_ball_position()
show(ball.get_ball())

# Preparing the pad
class Pad:
    def __init__(self, window_size=(525, 700, 3), num_pad = 1, position=(470, 270), 
                 lengths=(50, 200), color=[120, 100, 100]):
        """
            window_size: final window
            num_pad: number of pads
            position: top left point of pad
            lengths: lengths of pad on x, y dirn
            color: color of pad
                
        """
        self.window_size = window_size
        self.num_pad = num_pad
        self.position = position
        self.lengths = lengths
        self.color = color
        self.mid_point = (int(position[0]+self.lengths[0])/2, int(position[1]+self.lengths[1])/2)
         
    def init_pad(self):
        position = self.position
        lengths = self.lengths
        color = self.color
        pad_img = np.zeros(self.window_size).astype(np.uint8)
        pad_img[position[0]:position[0]+lengths[0], position[1]:position[1]+lengths[1]] = color
        #show(pad_img)
        return pad_img
    
    def get_pad(self, position):
        self.position = position       
        return self.init_pad()
pad = Pad()
show(pad.get_pad((400, 100)))
show(pad.get_pad((400, 300)))

# Prepare the Brick
class Brick:
    def __init__(self, size=(525, 700, 3), num_bricks=50, 
                 extreme_lengths=(10, 0), 
                 pbc=10, brick_part=40, bpr=10):
        """
            size: window size
            num_bricks: how many bricks total?
            extreme_lengths: minimum length and maximum length of single brick. Will be factored later.
            pbc: pixel per column
            brick_part: how many of the total size is used for brick region. 40 means 40% or rows from top.
            bpr: brick per row. 10 means 10 bricks will be used. 
        """
        self.num_bricks=num_bricks
        self.brick_part = int(brick_part/100 * size[0])
        self.extreme_lengths = (int(size[1]/bpr)-extreme_lengths[0], int(size[1]/bpr))
        self.brick_height = int(self.brick_part/bpr)
        self.bpr = bpr
        self.current_bricks = None
        self.size = size
        self.pbc = pbc
        self.init_bricks()
        
        
    def init_bricks(self):
        extreme_lengths = self.extreme_lengths
        num_bricks = self.num_bricks
        brick_height = self.brick_height
        
        bricks = np.random.randint(extreme_lengths[0], extreme_lengths[1], (int(num_bricks/self.bpr), self.bpr))
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 225, 0], [0, 255, 225], [255, 0, 255]])
        colors = colors[np.random.randint(0, len(colors), num_bricks)]
        brick_colors = {}
        bricks_group = []
        c=0
        for i, row in enumerate(bricks):
            for j, brick in enumerate(row):
                bricks_group.append([(brick_height*i, brick_height*(i+1)), (row[:j].sum(), row[:j].sum()+brick)])
                brick_colors[(brick_height*i, brick_height*(i+1), row[:j].sum(), row[:j].sum()+brick)] = colors[c]
                c+= 1
        bricks_group = np.array(bricks_group).reshape(bricks.shape[0], bricks.shape[1], -1)
        brick_img = np.zeros((bricks.shape[0]*brick_height, bricks.sum(axis=1).max(), 3))
        c = 0
        for row in bricks_group:
            for col in row:
                brick_img[col[0]:col[1], col[2]:col[3]] = brick_colors[(col[0], col[1], col[2], col[3])]
                c+=1
        brick_img = brick_img.astype(np.uint8)
        self.bricks_group = bricks_group
        self.colors = brick_colors
        self.current_bricks = self.bricks_group.tolist()
        self.current_brick_img = brick_img
        #print(self.colors)
        #show(brick_img)
        
        #return brick_img
    
    def set_bricks(self):
        return self.current_brick_img
    def check_hit(self, ball_pos, ball):
        score = 0
        new_bricks = []
        for i, rows in enumerate(self.current_bricks[::-1]):
            #print(brick)
            #break
            new_row=[]
            for j, brick in enumerate(rows):
                #print(brick, ball_pos)
                if brick[0]<=ball_pos[0]<brick[1] and brick[2]<=ball_pos[1]<brick[3]:
                    self.colors[(brick[0], brick[1], brick[2], brick[3])] = np.uint8([0, 0, 0])
                    self.current_brick_img[brick[0]:brick[1], brick[2]:brick[3]] = np.uint8([0, 0, 0])
                    p2 = (int((brick[0]+brick[1])/2), int((brick[2]+brick[3])/2))
                    ball.set_direction(p1=ball_pos, p2=p2, kind="h")
                    score += 1
                else:
                    new_row.append(brick)
            new_bricks.append(new_row)
        self.current_bricks = new_bricks
        return score
brick = Brick()
show(brick.current_brick_img)

# Game
class Game:
    def __init__(self,roi=None, game_window_size = (525, 700, 3), 
                 pad_position=(470, 270), lengths=(50, 200), avg_frames=100):
        """
            roi: required for contour region preparation
            game_window_size: window size
            pad_position: where should pad be initially?
            lengths: pad lengths on both coordinates
            avg_frames: how many frames to take running average?
        """
        self.pad_pointer = None
        self.roi = roi
        self.game_window_size = game_window_size
        self.pad_position = pad_position
        self.lengths = lengths
        self.ball = Ball(extreme_pos=((15, 15), (game_window_size[0]-15, game_window_size[1]-15)))
        self.pad = Pad()
        self.brick = Brick()
        self.score = 0
        self.avg_frames = avg_frames
    def get_game_window(self, pad_pointer):
        """
            Returns game window.
            pad_pointer: current position of pad
        """
        self.roi = self.roi
        game_window = np.zeros(self.game_window_size).astype(np.uint8)
        top, right, bottom, left = self.roi
        m = pad_pointer # c, r
        h = bottom - top
        l = left - right

        # use pad position instead of shape
        #pshape = (pad_pos[0][1]-pad_pos[0][0], pad_pos[1][1]-pad_pos[1][0])
        pshape = game_window.shape

        m = (int((m[0]/l)*pshape[1]), int((m[1]/h)*pshape[0]))    
        cv2.circle(game_window, (m[0], self.pad_position[0]), 5, [255, 255, 0], -3)

        # clip the position of pad only withing boundries
        position = (self.pad_position[0],  np.clip(m[0], 0, self.game_window_size[1]-self.lengths[1]))
        pad = self.pad.get_pad(position=position)
        
        bricks = self.brick.current_brick_img
        bshape = bricks.shape
        game_window[:bshape[0], :bshape[1]] = bricks
        
        ball = self.ball.get_ball()
        self.check_collision()
        window = game_window+pad+ball
        ball_stat = self.ball.ball_stat
        #print(self.game_stat)
        return window
    
    def check_collision(self):
        """
            Check collision with bricks/boundaries
        """
        ball = self.ball
        brick = self.brick
        pad = self.pad
        
        # check collision of pad and ball first
        if pad.position[0] <= ball.position[1]:
            #print(ball.position, pad.position)
            if pad.position[1]+pad.lengths[1] < ball.position[0] or ball.position[0] < pad.position[1]-pad.lengths[1]:
                self.game_stat = "over"
            if pad.position[1] <= ball.position[0] <= int((pad.position[0]+pad.lengths[1])):
                # ball has hitted the pad now bounce the ball
                ball.set_direction(p1=(ball.position[1], ball.position[0]), p2=pad.mid_point, direction=ball.direction, kind="h")
            
        elif ball.position[1] <= brick.brick_part:
            # check if ball pos lies within brick
            score = brick.check_hit((ball.position[1], ball.position[0]), ball)
            
            self.score += score
        
            
    def main(self):
        cam = cv2.VideoCapture(0)
        top, right, bottom, left = self.roi
        num_frames = 0
        avg_frames = self.avg_frames
        move_bg = None
        aweight = 0.5
        pointer_color = [100, 100, 100]
        rd = 5
        cd = 15
        game_window = None
#         game = Game(roi = [top, right, bottom, left])
        frames = 0
        self.game_stat = "pause"
        while True:
            # read the camera result
            (ret, frame) = cam.read()
            # if camera has read frame
            if ret:
                key = cv2.waitKey(1) & 0xFF
                frame = imutils.resize(frame, width=700)
                # flip to remove mirror effect
                frame = cv2.flip(frame, 1)
                # clone it to not mess with real frame
                clone = frame.copy()
                gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
                h, w = frame.shape[:2]
                move_gray = gray[top:bottom, right:left]
                move_gray = cv2.GaussianBlur(move_gray, (7, 7), 0)
                if num_frames<avg_frames:
                    move_bg = running_average(move_bg, move_gray, aweight)                
                    # put frame number on frame
                    cv2.putText(clone, str(num_frames), (100, 100),
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                    num_frames+=1
                    if num_frames > avg_frames-2:
                        self.game_stat = "play"
                if game_window is None:
                    game_window = self.get_game_window(pad_pointer = self.pad.mid_point)
                else:
                    
                    move_hand = get_contours(move_bg, move_gray)
                    if move_hand is not None:
                        mthresholded, msegmented = move_hand
                        sshape = msegmented.shape
                        new_segmented = msegmented.reshape(sshape[0], sshape[-1])
                        m = new_segmented.min(axis=0)
                        cv2.drawContours(clone, [msegmented+(right, top)], -1, (0, 0, 255))
                        cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)
                        cv2.rectangle(clone, (int((right+m[0]))-cd, int((top+bottom)/2)-rd), 
                                      (int((right+m[0]))+cd, int((top+bottom)/2)+rd), (255, 0, 0), 2)
                        
                        if self.game_stat == "play":
                            game_window = game.get_game_window(pad_pointer = m)
                if game_window is not None:
                    if self.game_stat != "over":
                        cv2.putText(game_window, str(self.score), (10, 500),
                                           cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
                        if self.game_stat == "pause":
                            cv2.putText(game_window, str(self.game_stat), (250, 400),
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        frames+=1
                    elif self.game_stat == "over":
                        cv2.putText(game_window, "Over Score: " + str(self.score), (30, 300),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow("Break the Brick", game_window)
                    
                cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow("Running Frame", clone)
                if key==27:
                    break
        cam.release()
        cv2.destroyAllWindows()
game = Game(roi = [400, 400, 500, 681])
game.main()

