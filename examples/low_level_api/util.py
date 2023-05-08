
ANSI_COLOR_RESET = "\x1b[0m"
ANSI_COLOR_YELLOW = "\x1b[33m"
ANSI_BOLD = "\x1b[1m"
ANSI_COLOR_GREEN = "\x1b[32m"

CONSOLE_COLOR_DEFAULT = ANSI_COLOR_RESET
CONSOLE_COLOR_PROMPT = ANSI_COLOR_YELLOW
CONSOLE_COLOR_USER_INPUT = ANSI_BOLD + ANSI_COLOR_GREEN

# Iterative search
# Actively searches and prevents a pattern from being returned
class IterSearch:
	def __init__(self, pattern):
		self.pattern = list(pattern)
		self.buffer = []

	def __call__(self, char):
		self.buffer += [char]

		if (self.pattern[:len(self.buffer)] == self.buffer):
			if (len(self.buffer) >= len(self.pattern)):
				self.buffer.clear()
			return []

		_tmp = self.buffer[:]
		self.buffer.clear()
		return _tmp

class Circle:
	def __init__(self, size, default=0):
		self.list = [default] * size
		self.maxsize = size
		self.size = 0
		self.offset = 0

	def append(self, elem):
		if self.size < self.maxsize:
			self.list[self.size] = elem
			self.size += 1
		else:
			self.list[self.offset] = elem
			self.offset = (self.offset + 1) % self.maxsize

	def __getitem__(self, val):
		if isinstance(val, int):
			if 0 > val or val >= self.size:
				raise IndexError('Index out of range')
			return self.list[val] if self.size < self.maxsize else self.list[(self.offset + val) % self.maxsize]
		elif isinstance(val, slice):
			start, stop, step = val.start, val.stop, val.step
			if step is None:
				step = 1
			if start is None:
				start = 0
			if stop is None:
				stop = self.size
			if start < 0:
				start = self.size + start
			if stop < 0:
				stop = self.size + stop

			indices = range(start, stop, step)
			return [self.list[(self.offset + i) % self.maxsize] for i in indices if i < self.size]
		else:
			raise TypeError('Invalid argument type')




if __name__ == "__main__":
	c = Circle(5)

	c.append(1)
	print(c.list)
	print(c[:])
	assert c[0] == 1
	assert c[:5] == [1]

	for i in range(2,5+1):
		c.append(i)
	print(c.list)
	print(c[:])
	assert c[0] == 1
	assert c[:5] == [1,2,3,4,5]

	for i in range(5+1,9+1):
		c.append(i)
	print(c.list)
	print(c[:])
	assert c[0] == 5
	assert c[:5] == [5,6,7,8,9]
	#assert c[:-5] == [5,6,7,8,9]
	assert c[:10] == [5,6,7,8,9]

