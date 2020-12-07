from PySide2.QtWidgets import *
from PySide2.QtGui import QPainter, QPen, QBrush, QPixmap, QColor
from PySide2.QtCore import Qt
import sys
import numpy as np
from rb import Restricted_Boltzman

class Grader(QWidget):

	def __init__(self):
		super().__init__()
		self.setWindowTitle("arg__1")
		# field sizes
		self.x_amount = 28
		self.y_amount = 28
		self.cell_width = 20
		self.cell_height = 20
		# resize the window to the size of the field
		self.resize(self.cell_width*self.x_amount, self.cell_height*self.y_amount)
		# lists for full and empty rectangles
		self.full_rects = []
		self.empty_rects = []
		
		self.delete = False
		self.drawn = set()
		
		self.draw_field()
		self.table = np.zeros(shape=(28,28))
		print(self.table.size)

		self.rb = Restricted_Boltzman(0.1, 794, 794)
		self.rb.load_and_train(training_amount=10000)

	def resizeEvent(self, event):
		super().resizeEvent(event)
		# get the new dimensions
		geom = self.geometry()
		# adjust the cell height and width
		self.cell_height = geom.height()//self.y_amount			
		self.cell_width = geom.width()//self.x_amount
		# if the window size doesn't match up with the grid resize the window to match
		if geom.width() != self.x_amount * self.cell_width or geom.height() != self.y_amount * self.cell_height:
			self.resize(self.cell_width*self.x_amount, self.cell_height*self.y_amount)


	def draw_field(self,x_squares=28, y_squares=28):
		"""
		initially generates the field 
		"""
		# generate x_squares * y_squares entries for empty rects
		for i in range(y_squares):
			for j in range(x_squares):
				self.empty_rects.append((j, i))
		# draw the newly generated rectangles
		self.update()

	# overriden paint event
	def paintEvent(self,e):
		painter = QPainter(self)
		self.paint_rectangles(painter)

	def paint_rectangles(self, painter):
		"""
		Paints all rectangles from self.empty_rects and self.full_rects to the window
		the lists contain tuples x,y with x meaning this is the x'th rectangle in this row
		and y meaning the y't rectangle in the column 
		Calculate the coordinates of the top left corner by multiplying x with cell_width and y with cell_height

		params:
			painter: QPainter used to draw the rectangles
		"""
		# paint all empty rectangles white
		painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
		# paint all empty rectangles
		for rect in self.empty_rects:
			painter.drawRect(rect[0] * self.cell_width, rect[1]*self.cell_height, self.cell_width, self.cell_height)
		# paint all full rectangles black
		painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
		# paint all full rectangles
		for rect in self.full_rects:
			painter.drawRect(rect[0] * self.cell_width, rect[1]*self.cell_height, self.cell_width, self.cell_height)
		# close the painter
		painter.end()


	def mouseMoveEvent(self, event):
		"""
		Overriden mouse move event method
		this method is only called when the mouse is moved, while a mouse button is pressed
		this method updates the rectangle the mouse is currently over according to the
		pressed mouse button(left=draw, right=delete)
		"""
		# get the mouse position
		pos = event.pos()
		# get the rectangle the mouse is currently over
		rect = self.get_current_rectangle(pos.x(), pos.y())

		# if the rectangle hasn't already been drawn draw it
		if rect not in self.drawn:
			# if delete is active move the rectangle to the empty rectangle list
			if self.delete:
				# check if the rect is in the full rect list and remove it
				if rect in self.full_rects:
					self.full_rects.remove(rect)
				# add the rect to the empty rect list
				self.empty_rects.append(rect)
				# adjust the value in the table 
				# rect[0] is the x coordinate which correspond to a column in the table and
				# rect[1] is the y coordinate which corresponds to a row in the table
				self.table[rect[1], rect[0]] = 0
			else:
				# check if the rect is in the empty rect list and remove it
				if rect in self.empty_rects:
					self.empty_rects.remove(rect)
				# add the rect to the full rect list
				self.full_rects.append(rect)
				# adjust the value in the table
				self.table[rect[1], rect[0]] = 1
			# add the rect to the already drawn set
			self.drawn.add(rect)
			# display the changes 
			self.update()
			# identify the drawing
			print(self.rb.identify(self.table.flatten(), True))

	def mousePressEvent(self, event):
		"""
		Overriden mousepress event listener
		this method dictates which action is taken when a mouse button is pressed
		LMB = Draw, RMB = Delte current rectangle, MB3 = clear field
		"""
		if event.button() == Qt.MouseButton.LeftButton:
			self.delete = False
		elif event.button() == Qt.MouseButton.RightButton:
			self.delete = True
		else:
			# clear the field
			# add all rects to the empty rects list
			self.empty_rects.extend(self.full_rects)
			# empty the full rects list and adjust the values in the table
			self.full_rects = []
			self.table *= 0

	def mouseReleaseEvent(self, event):
		# reset the already drawn set upon release of the mouse button
		self.drawn = set()

	def get_current_rectangle(self, x, y):
		x = x // self.cell_width
		y = y // self.cell_height
		return (x,y)

class Rectangle(QLabel):
	def __init__(self, x, y, width, height):
		super().__init__()
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.pixmap


	

if __name__ == "__main__":
	app = QApplication([])
	widget = Grader()
	widget.show()
	sys.exit(app.exec_())