


all: t2clean t2


t1:
	cd task1_cart_polar/ && ../../openfast/build-single-debug/glue-codes/fast-farm/FAST.Farm input_cart.fstf
	cd task1_cart_polar/ && ../../openfast/build-single-debug/glue-codes/fast-farm/FAST.Farm input_polar.fstf
	python paperFigures.py t1
#	cd task1_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf


t1c:
	cd task1c_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_cart.fstf
	cd task1c_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf
	cd task1c_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_polar.fstf
	/Users/lmartin1/opt/anaconda3/bin/python paperFigures.py t1c

t1b:
	cd task1b_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_polar.fstf
	cd task1b_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf
	cd task1b_cart_polar/ && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_cart.fstf
	/Users/lmartin1/opt/anaconda3/bin/python paperFigures.py t1b

y:
	cd yawstudy && ../../openfast_emmanuel/build/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
yr:
	cd yawstudy && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
	
t2:
	cd task2_adm && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
	/Users/lmartin1/opt/anaconda3/bin/python paperFigures.py t2
	
t2clean:
	rm -rf task2_adm/vtk_ff_planes
	rm -rf task2_adm/vtk_ff	
	
t3:
	cd task3_alm && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
	/Users/lmartin1/opt/anaconda3/bin/python paperFigures.py t3
	
	
t3clean:
	rm -rf task3_alm/vtk_ff_planes
	rm -rf task3_alm/vtk_ff	
	
	
t4:
	cd task4_transient && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_polar.fstf 
	cd task4_transient && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
#	cd task4_transient && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl_coarse.fstf 
	/Users/lmartin1/opt/anaconda3/bin/python paperFigures.py t4
	
	
t4clean:
	rm -rf task4_transient/vtk_ff_planes
	rm -rf task4_transient/vtk_ff	
	
	
	
	
	
	
	
t5:
	cd task5_secondary && ../../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm input_curl.fstf 
	
	
t5clean:
	rm -rf task5_secondary/vtk_ff_planes
	rm -rf task5_secondary/vtk_ff	
	
	
	
	
	
	
	
	
pcr:
	../openfast_emmanuel/build-release/glue-codes/fast-farm/FAST.Farm yawconvection/input_polar_convect_bigger.fstf 
		
clean:
	rm -rf vtk_ff_plane
	rm -rf vtk_ff
