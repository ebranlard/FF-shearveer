


all: t1


t1:
	cd task1_uniform_aligned/ && ../../openfast/build-linux-single-debug/glue-codes/fast-farm/FAST.Farm input_cart.fstf
	cd task1_uniform_aligned/ && ../../openfast/build-linux-single-debug/glue-codes/fast-farm/FAST.Farm input_polar.fstf
	python task1.py
		
clean:
	rm -rf task1_uniform_aligned/vtk_ff_plane
	rm -rf task1_uniform_aligned/vtk_ff
