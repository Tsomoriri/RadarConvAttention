from src.process_data.synthetic_data import synData

print('hello')
syn_data = synData(x=40, y=40, t=20, pde='advection', mux=0.5, muy=0.5, ictype='3rect',n_blobs=3)
rect_movie = syn_data.generate_rectangles_with_individual_velocities(3,40,40,20)
syn_data.save_movie(rect_movie, '3rect_movie.npy')
