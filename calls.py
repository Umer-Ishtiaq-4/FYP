import matplotlib.pyplot as plt
import generate_pos_process as instance
import pprint
import processed_input as process_input
import pretrained as pt



input_text = "The building contain two bedrooms and a livingroom. livingroom1 is connected to bedroom2. bedroom2 is connected to bedroom1."

information_extracted = process_input.process_input(input_text)
# information_extracted = pt.process_input(input_text)

information_extracted = process_input.check_sizes(information_extracted)
pprint.pprint(information_extracted)
im = instance.Generate(information_extracted)
if str(type(im)) == "<class 'PIL.Image.Image'>":
    # print(type(im))
    print("\nImage in view...")
    plt.imshow(im)
    plt.show()
    print("\nImage Closed!")

else:
    print("\n**Returned Object is not valid!!!**")



# information_extracted = {'links': [['livingroom1', 'bedroom1'],
#     ['livingroom1', 'bedroom2'],
#     ['livingroom1', 'bedroom3'],
#     ['bedroom1', 'bedroom2'],
#     ['bedroom1', 'bedroom3']],
#     'rooms': ['bedroom1', 'bedroom2', 'bedroom3', 'livingroom1'],
#     'sizes': {'bedroom1': [0, 'NW'],
#     'bedroom2': [0, 'N'],
#     'bedroom3': [None, 'NE'],
#     'livingroom1': [30, 'C']}}