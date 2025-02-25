'''
 This is going to be the main driver file: Handles user input and displays the current GameState object.
'''

import pygame as p
# we are calling the parent folder 'Chess,' the name which I have stored in drive
# include 'form parent folder name' before this
import ChessEngine

WIDTH = HEIGHT = 512 #400 is another option to be used for (based on the current images)
# Note this is for the board and going that going any higher might sacrifice resolution

DIMENSION = 8 #8x8

SQ_SIZE = HEIGHT // DIMENSION # just calculating the NxN value for the size of each square.
# We use floor division here because we want to ensure that all the squares fit perfectly in the board.

MAX_FPS = 15 #Used for animations later on

IMAGES = {}

'''
We want to ensure we only load in the images ONCE, this is an expensive operation and will sacrifice a lot of memory and time.
So we intiialize a global dictionary of images. 
'''

def loadImages():
    #Utilize a for loop to speed up this process, note that the order of the dictionary does not matter, this is a process that will be done ONLY ONCE
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))
    # this gives us the ability to call the image for any piece as such: IMAGES['wp']

    '''
    The following will be our main driver, which handles the user input as well as the main graphics
    '''

def main():
        p.init()
        screen = p.display.set_mode((WIDTH, HEIGHT))
        # Note that this is NOT the game clock, it just controls the game responsiveness timing
        clock = p.time.Clock()
        screen.fill(p.Color("white"))
        gs = ChessEngine.GameState()
        print(gs.board)
        loadImages()#Only do this once
        running = True
        while running:
            for e in p.event.get():
                # event.get() essentially is the most recent keystroke, button press, etc. that the user makes
                # so here if the user decides to quit the application, we make running = false.
                if e.type == p.QUIT:
                    running = False
            # limits the game's frame rate to whatever we set this max to be
            drawGameState(screen, gs)
            clock.tick(MAX_FPS)
            # provides visual update
            p.display.flip()


# This function is reponsible for the main display of the current game state.         
def drawGameState(screen, gs):
     drawBoard(screen) # draws the squares on the board
     drawPieces(screen, gs.board) #draw pieces on the squares
     
def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    # here we consider this, in a chess board, for ex a matrix that is 8x8, when you add the dimensions, for ex. (0,0) or (5,5) --> 0+0=0 or 5+5=10, you will see
    # that the even numbers and odd numbers correspond to white or black respectively. This way we can tell which square should be what color, which is very helpful in our nested loop
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            # if we get an even value, then our result will be 0, making that square white
            # An interesting way to implement color: 
            color = colors[((r+c)%2)]
            # c is technically our 'x' value and r is 'y'
            # the first two parameters in Rect the location of the square and the next two are the size dimensions
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
# Note that the contents of this function could technically be placed in the function above it, however, if we want to add, let's say, piece highlighting
# it makes it easier to separate the drawing of the pieces and the actual squares themselves. 
def drawPieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != "--": #If the piece is not an empty square, we want to draw it on here
                # blit function allows you to draw one image on top of the other. 
                screen.blit(IMAGES[piece], p.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

if __name__ == "__main__":
     # Python recommends doing this, as we dont have to be in the current file to run the main function
     main()