import pygame
import tensorflow as tf
import numpy as np

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 18)


class Cell:
    def __init__(self, pos, value):
        self.size = 20
        self.cellValue = value
        self.pos = pos


    def createCell(self, surface):
        pygame.draw.rect(surface, (0,0,0), (self.pos[0], self.pos[1], self.size, self.size))

    def drawCell(self, surface, buttonSelected):
        if buttonSelected == "Draw":
            if self.cellValue:
                pygame.draw.rect(surface, (255,255,255), (self.pos[0], self.pos[1], self.size, self.size))
            else:
                pygame.draw.rect(surface, (0,0,0), (self.pos[0], self.pos[1], self.size, self.size))




class Grid:

    def __init__(self):
        self.cells = []

        for y in range(28): 
            self.cells.append([])
            for x in range(28):
                self.cells[y].append(Cell((x*20, y*20),False))

        self.lines = []

        for y in range(1, 29, 1): 
            temp = []
            temp.append((0, y * 20))
            temp.append((560, y * 20))
            self.lines.append(temp)

        for x in range(1, 29, 1):
            temp = []
            temp.append((x*20, 0))
            temp.append((x*20, 560))
            self.lines.append(temp)



    # This method recreates the lines after a cell is modified
    def redrawLines(self,x,y,surface):
        pygame.draw.line(surface, (0,125,0), self.lines[y-1][0], self.lines[y-1][1])
        pygame.draw.line(surface, (0,125,0), self.lines[(x+28)-1][0], self.lines[(x+28)-1][1]) 

    # This method creates all the cells in the grid
    def draw(self, surface):
        for row in self.cells:
            for cell in row:
                cell.createCell(surface)
        for line in self.lines:
            pygame.draw.line(surface, (0, 125, 0), line[0], line[1])


    # Check whether x and y are within the bounds of the grid
    def isWithinBounds(self, x, y):
        return x >= 0 and x < 28 and y >= 0 and y < 28


    # Draw a cell and all neighboring cells
    def drawCell(self, x, y, surface, buttonSelected):
        if not self.isWithinBounds(x,y):
                return

        def drawNeighboringCells(y,x):
            if not self.isWithinBounds(x,y):
                return
            self.cells[y][x].cellValue = True
            self.cells[y][x].drawCell(surface, buttonSelected)
            self.redrawLines(x,y,surface)
        
        # Current Cell
        try:
            if not self.cells[y][x].cellValue:
                drawNeighboringCells(y, x)
        except Exception:
            pass

        # Up
        try:
            if not self.cells[y-1][x].cellValue:
                drawNeighboringCells(y-1, x)
        except Exception:
            pass

        # Down
        try:
            if not self.cells[y+1][x].cellValue:
                drawNeighboringCells(y+1, x)
        except Exception:
            pass

        # Left
        try:
            if not self.cells[y][x-1].cellValue:
                drawNeighboringCells(y, x-1)
        except Exception:
            pass

        # Right
        try:
            if not self.cells[y][x+1].cellValue:
                drawNeighboringCells(y, x+1)
        except Exception:
            pass

    


    # This method predicts the number the user has drawn
    def predict(self, surface, class_names, interpreter, input_index, output_index):

        # Create a test set to predict
        test = np.zeros((28,28,1), dtype=float)
        for row in range(28):
            for cell in range(28):
                if self.cells[row][cell].cellValue:     
                    num = 1.0; num = tf.cast(num, tf.float32)
                    test[row][cell][0] = num
                else:
                    num = 0.0; num = tf.cast(num, tf.float32)
                    test[row][cell][0] = num
                
        
        test = np.expand_dims(test, axis=0).astype(np.float32)
        image_tensor = np.vstack([test])
        interpreter.set_tensor(input_index, image_tensor)
        interpreter.invoke()
        return class_names[np.argmax(interpreter.get_tensor(output_index)[0])]

    
    # This method reloads the window
    def reload(self, surface):
        for row in range(28):
            for cell in range(28):
                if self.cells[row][cell].cellValue == True:
                    self.cells[row][cell].cellValue = False
                    self.cells[row][cell].drawCell(surface, 'Draw')
                    self.redrawLines(cell,row,surface)

        self.cells.clear
