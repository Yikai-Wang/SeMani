import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self, suffix=""):
        """save the current content to the HMTL file"""
        html_file = '%s/index%s.html' % (self.web_dir, suffix)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def display_current_results(save_dir, begin_epoch, current_epoch, sentence_dataset_id_list, img_size=256, img_path_pattern="./Epoch%d/%d.jpg", title=""):
    """Display current results on visdom; save current results to an HTML file.

    Parameters:
        visuals (OrderedDict) - - dictionary of images to display or save
        epoch (int) - - the current epoch
        save_result (bool) - - if save the current results to an HTML file
    """
    # update website
    webpage = HTML(save_dir, title=title, refresh=0)
    for sentence, dataset, id in sentence_dataset_id_list:
        webpage.add_header(sentence)
        ims, txts, links = [], [], []
        for n in range(begin_epoch, current_epoch + 1):
            img_path = img_path_pattern % (n, id)
            ims.append(img_path)
            txts.append("Epoch_%d" % n)
            links.append(img_path)
        webpage.add_images(ims, txts, links, width=img_size)
    webpage.save(suffix="_%s" % sentence_dataset_id_list[0][1])


if __name__ == '__main__':  # we show an example usage here.
    sentence_dataset_id_list = []
    for i in range(10):
        sentence_dataset_id_list.append(["sentence_%d" % i, "coco", i])
    display_current_results("./", 30, sentence_dataset_id_list, title="fdjklaghfeuioprtoalfk")
