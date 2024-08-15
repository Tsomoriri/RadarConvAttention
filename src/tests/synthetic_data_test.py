


from src.process_data.synthetic_data import synData







# Generate and save 3rect movie
syn_data = synData(
    x=40, y=40, t=20, pde="advection", mux=0.5, muy=0.5, ictype="3rect", n_blobs=3
)
rect_movie = syn_data.generate_rectangles_with_individual_velocities(3, 40, 40, 20)
syn_data.save_movie(rect_movie, "3rect_movie.npy")
# rect_movie = syn_data.generate_ill_movie()
# syn_data.save_movie(rect_movie, '3ill_movie.npy')

# # Generate and save 11rect movie
# syn_data = synData(x=40, y=40, t=20, pde='advection', mux=0.5, muy=0.5, ictype='3rect',n_blobs=11)
# rect_movie = syn_data.generate_movie()
# syn_data.save_movie(rect_movie, '11rect_movie.npy')
# rect_movie = syn_data.generate_ill_movie()
# syn_data.save_movie(rect_movie, '11ill_movie.npy')
