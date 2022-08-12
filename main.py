import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import RAISED, filedialog
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
from matplotlib.ft2font import BOLD
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

_DEBUG = False

# Center tkinter window
def centerWindow(app):
    app.update_idletasks()
    width = app.winfo_width()
    frm_width = app.winfo_rootx() - app.winfo_x()
    app_width = width + 2 * frm_width

    height = app.winfo_height()
    titlebar_height = app.winfo_rooty() - app.winfo_y()
    app_height = height + titlebar_height + frm_width

    pos_x = app.winfo_screenwidth() // 2 - app_width // 2
    pos_y = app.winfo_screenheight() // 2 - app_height // 2
    app.geometry('{}x{}+{}+{}'.format(width, height, pos_x, pos_y - 50))
    app.deiconify()

class ClusterAnalysisApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.title(self, "Cluster Analysis Tool")

        container = tk.Frame(self)
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        self.frames = {}
        for F in (StartPage, KMeans_Page1, KMeans_Page2, HRCHCL_Page1, HRCHCL_Page2, DBSCAN_Page1, DBSCAN_Page2):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky = "nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # Initialize Canvas
        canvas = tk.Canvas(self, width=920, height=650, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # Labels & Image
        label1 = tk.Label(self, text="Welcome to Cluster Analysis Tool.", font = ("Helvetica", 35))
        canvas.create_window(460, 50, window = label1)

        label2 = tk.Label(self, text="Visualize, plot, and cluster your data! Pick any of the 3 available algorithms below to begin.", font = ("Helvetica", 15))
        canvas.create_window(460, 120, window = label2)

        image1 = Image.open("clusterpic.png")
        image1 = image1.resize((200, 200), Image.LANCZOS) # Resampling.LANCZOS (2023-07-01)
        test = ImageTk.PhotoImage(image1)
        picLabel = tk.Label(self, image=test)
        picLabel.image = test
        canvas.create_window(460, 300, window = picLabel)

        label3 = tk.Label(self, text="Choose Your Desired Algorithm:", font = ("Helvetica", 15))
        canvas.create_window(460, 480, window = label3)

        # Navigate to K-Means button
        button1 = ttk.Button(self, text="K-Means", command=lambda: controller.show_frame(KMeans_Page1))
        canvas.create_window(460, 520, window = button1)

        # Navigate to Hierarchical button
        button2 = ttk.Button(self, text="Hierarchical", command=lambda: controller.show_frame(HRCHCL_Page1))
        canvas.create_window(460, 545, window = button2)

        # Navigate to DBSCAN button
        button3 = ttk.Button(self, text="DBSCAN", command=lambda: controller.show_frame(DBSCAN_Page1))
        canvas.create_window(460, 570, window = button3)


class KMeans_Page1(tk.Frame):    # k-means page
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=400, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # Label: k-means algorithm title
        headingLabel = tk.Label(self, text = "k-means algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = headingLabel)   

        # Entry: chooseColumnsEntry
        def placeholder_text(e):
            chooseColumnsEntry.delete(0, "end")
        def placeholder_text2(e):
            numOfClustersEntry.delete(0, "end")

        global chooseColumnsEntry, numOfClustersEntry
        chooseColumnsEntry = tk.Entry(self, width = 20)
        chooseColumnsEntry.insert(0, "Choose columns to plot...")
        chooseColumnsEntry.bind("<FocusIn>", placeholder_text)
        numOfClustersEntry = tk.Entry(self)
        numOfClustersEntry.insert(0, "Enter number of clusters...")
        numOfClustersEntry.bind("<FocusIn>", placeholder_text2)
        
        # Functions
        def getImportData():
            global df
            import_file_path = filedialog.askopenfilename()
            df = pd.read_csv(import_file_path)

            if(_DEBUG == True):
                print(df.head())
                print("Row Count: " + str(len(df)), end = "\n")
                print("Col Count: " + str(len(df.axes[1])), end = "\n")
            
            global colNamesString
            colNamesString = ""
            for i in range(len(df.axes[1])):
                colNamesString += str(df.columns[i]) if(i == len(df.axes[1]) - 1) else str(df.columns[i]) + ", "
            if(_DEBUG == True): print("colString: " + colNamesString + '\n')

            # Column Names Label
            colNamesLabel = tk.Label(self, text = "Columns: " + colNamesString, font = ("Times", 14))
            canvas.create_window(325, 160, window = colNamesLabel)

            # Number of Columns Label
            rowCountLabel = tk.Label(self, text = "Row Count: " + str(len(df.axes[0])), font = ("Times", 14))
            canvas.create_window(325, 190, window = rowCountLabel)

            if(len(df.axes[1]) <= 2):
                canvas.create_window(325, 295, window = numOfClustersEntry)
            elif(len(df.axes[1]) >= 3):
                canvas.create_window(325, 280, window = chooseColumnsEntry)
                canvas.create_window(325, 310, window = numOfClustersEntry)
                if(len(df.axes[1]) >= 4):
                    PCA_notif_label = tk.Label(self, text = "*PCA will be used for dimension reduction (4+ col)", font = ("Times", 10))
                    canvas.create_window(325, 340, window = PCA_notif_label)

            # Next Button & Num of Clusters entry
            GoToKMeansP2Button = tk.Button(self, text = "Next", command=lambda: controller.show_frame(KMeans_Page2), bg = "blue", relief = RAISED)
            canvas.create_window(325, 400, window = GoToKMeansP2Button)
                
        # Import Excel Data Button
        browseButtonExcel = tk.Button(self, text = "Import Data (.csv)", command = getImportData, bg = "blue", relief = RAISED)
        canvas.create_window(325, 80, window = browseButtonExcel)

        # Back to Home button
        backToHomeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        backToHomeButton.pack(side = tk.BOTTOM, pady = 10)


class KMeans_Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=120, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # k-means algorithm title label
        label1 = tk.Label(self, text = "k-means algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = label1)   
        
        def runKMeans():
            userChoiceArray = list(df.columns) if(len(df.axes[1]) == 2) else chooseColumnsEntry.get().replace(" ", "").split(',')
            if(_DEBUG == True): print("USER_CHOICE_ARRAY: " + str(userChoiceArray) + "\n")
            numberOfClusters = int(numOfClustersEntry.get())
            figure = plt.Figure(figsize = (4,3), dpi = 100)

            # Get centroids of data using kmeans
            if(len(userChoiceArray) == 2 or len(userChoiceArray) == 3):
                kmeans = KMeans(n_clusters = numberOfClusters).fit(df[userChoiceArray])
                centroids = kmeans.cluster_centers_
                
            if(len(userChoiceArray) == 2):
                ax = figure.add_subplot(111)
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], c = kmeans.labels_.astype(float), s = 80, alpha = 0.5)
                ax.scatter(centroids[:, 0], centroids[:, 1], c = "red", s = 50)

                # Set axes names
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)
                
            elif(len(userChoiceArray) == 3):
                ax = figure.add_subplot(111, projection='3d')
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], df[userChoiceArray[2]], c = kmeans.labels_.astype(float), s = 80, alpha = 0.5)
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c = "red", s = 50)

                # Set axes names (3d)
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)
                ax.set_zlabel(userChoiceArray[2], fontsize = 10) 
            
            elif(len(userChoiceArray) >= 4):
                # Dim reduction using PCA
                x = StandardScaler().fit_transform(df[userChoiceArray])
                pca = PCA(n_components = 2)
                principal_components = pca.fit_transform(x)
                principal_df = pd.DataFrame(data = principal_components, columns = ['pca1', 'pca2'])

                # Get centroids of data using kmeans
                kmeans = KMeans(n_clusters = numberOfClusters).fit(principal_df)
                centroids = kmeans.cluster_centers_

                ax = figure.add_subplot(111)
                ax.scatter(principal_df["pca1"], principal_df["pca2"], c = kmeans.labels_.astype(float), s = 80, alpha = 0.5)
                ax.scatter(centroids[:, 0], centroids[:, 1], c = "red", s = 50)
                ax.set_xlabel("Principal Component 1", fontsize = 10)
                ax.set_ylabel("Principal Component 2", fontsize = 10)

            # Pack figure
            scatter1 = FigureCanvasTkAgg(figure, self)
            scatter1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            toolbar = NavigationToolbar2Tk(scatter1, self)
            toolbar.update()
            scatter1._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
           
        # Buttons
        # Back to KMEANS Page 1 button
        backButton = ttk.Button(self, text = "Back", command = lambda: controller.show_frame(KMeans_Page1))
        canvas.create_window(0, 45, window = backButton)

        # Process Button
        processButton = tk.Button(self, text = "Run k-means ▶", command = runKMeans)
        canvas.create_window(325, 80, window = processButton)

        # Back to Home button
        homeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        homeButton.pack(side = tk.BOTTOM, pady = 10)


class HRCHCL_Page1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=400, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # Label: k-means algorithm title
        headingLabel = tk.Label(self, text = "hierarchical algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = headingLabel)   

        # Entry: chooseColumnsEntry
        def placeholder_text(e):
            chooseColumnsEntry2.delete(0, "end")
        def placeholder_text2(e):
            numOfClustersEntry2.delete(0, "end")

        global chooseColumnsEntry2, numOfClustersEntry2
        chooseColumnsEntry2 = tk.Entry(self, width = 20)
        chooseColumnsEntry2.insert(0, "Choose columns to plot...")
        chooseColumnsEntry2.bind("<FocusIn>", placeholder_text)
        numOfClustersEntry2 = tk.Entry(self)
        numOfClustersEntry2.insert(0, "Enter number of clusters...")
        numOfClustersEntry2.bind("<FocusIn>", placeholder_text2)

        # Functions
        def getImportData():
            global df
            import_file_path = filedialog.askopenfilename()
            df = pd.read_csv(import_file_path)

            if(_DEBUG == True):
                print(df.head())
                print("Row Count: " + str(len(df)), end = "\n")
                print("Col Count: " + str(len(df.axes[1])), end = "\n")
            
            global colNamesString
            colNamesString = ""
            for i in range(len(df.axes[1])):
                colNamesString += str(df.columns[i]) if(i == len(df.axes[1]) - 1) else str(df.columns[i]) + ", "
            if(_DEBUG == True): print("colString: " + colNamesString + '\n')

            # Column Names Label
            colNamesLabel = tk.Label(self, text = "Columns: " + colNamesString, font = ("Times", 14))
            canvas.create_window(325, 160, window = colNamesLabel)

            # Number of Columns Label
            rowCountLabel = tk.Label(self, text = "Row Count: " + str(len(df.axes[0])), font = ("Times", 14))
            canvas.create_window(325, 190, window = rowCountLabel)

            if(len(df.axes[1]) <= 2):
                canvas.create_window(325, 295, window = numOfClustersEntry2)
            elif(len(df.axes[1]) >= 3):
                canvas.create_window(325, 280, window = chooseColumnsEntry2)
                canvas.create_window(325, 310, window = numOfClustersEntry2)
                if(len(df.axes[1]) >= 4):
                    PCA_notif_label = tk.Label(self, text = "*PCA will be used for dimension reduction (4+ col)", font = ("Times", 10))
                    canvas.create_window(325, 340, window = PCA_notif_label)

            # Next Button
            GoToHRCHCLP2Button = tk.Button(self, text = "Next", command=lambda: controller.show_frame(HRCHCL_Page2), bg = "blue", relief = RAISED)
            canvas.create_window(325, 400, window = GoToHRCHCLP2Button)
                
        # Import Excel Data Button
        browseButtonExcel = tk.Button(self, text = "Import Data (.csv)", command = getImportData, bg = "blue", relief = RAISED)
        canvas.create_window(325, 80, window = browseButtonExcel)

        # Back to Home button
        backToHomeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        backToHomeButton.pack(side = tk.BOTTOM, pady = 10)


class HRCHCL_Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=120, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # k-means algorithm title label
        label1 = tk.Label(self, text = "hierarchical algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = label1)   

        def showDendrogram():
            userChoiceArray = list(df.columns) if(len(df.axes[1]) == 2) else chooseColumnsEntry2.get().replace(" ", "").split(',')
            clusters = linkage(df[userChoiceArray], method='ward', metric='euclidean')
            dendrogram(clusters)
            plt.show()

        def runHRCHCL():
            userChoiceArray = list(df.columns) if(len(df.axes[1]) == 2) else chooseColumnsEntry2.get().replace(" ", "").split(',')
            figure = plt.Figure(figsize = (4,3), dpi = 100)

            if(len(userChoiceArray) == 2 or len(userChoiceArray) == 3): 
                agglo = AgglomerativeClustering(n_clusters=int(numOfClustersEntry2.get()), affinity='euclidean', linkage='ward').fit(df[userChoiceArray])

            if(len(userChoiceArray) == 2):
                ax = figure.add_subplot(111)
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], c = agglo.labels_, s = 80, alpha = 0.5)
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)

            elif(len(userChoiceArray) == 3):
                ax = figure.add_subplot(111, projection='3d')
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], df[userChoiceArray[2]], c = agglo.labels_.astype(float), s = 80, alpha = 0.5)
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)
                ax.set_zlabel(userChoiceArray[2], fontsize = 10)

            elif(len(userChoiceArray) >= 4):
                x = StandardScaler().fit_transform(df[userChoiceArray])
                pca = PCA(n_components = 2) 
                principal_components = pca.fit_transform(x)
                principal_df = pd.DataFrame(data = principal_components, columns = ['pca1', 'pca2'])

                agglo = AgglomerativeClustering(n_clusters=int(numOfClustersEntry2.get()), affinity='euclidean', linkage='ward').fit(principal_df)
                ax = figure.add_subplot(111)
                ax.scatter(principal_df["pca1"], principal_df["pca2"], c = agglo.labels_.astype(float), s = 80, alpha = 0.5)
                ax.set_xlabel("Principal Component 1", fontsize = 10)
                ax.set_ylabel("Principal Component 2", fontsize = 10)

            # Pack figure
            scatter1 = FigureCanvasTkAgg(figure, self)
            scatter1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            toolbar = NavigationToolbar2Tk(scatter1, self)
            toolbar.update()
            scatter1._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
           
        # Buttons
        # Process Button
        showDendroButton = tk.Button(self, text = "Show Dendrogram", command = showDendrogram)
        canvas.create_window(325, 80, window = showDendroButton)
        
         # Process Button
        processButton = tk.Button(self, text = "Run Hierarchical ▶", command = runHRCHCL)
        canvas.create_window(325, 110, window = processButton)

        # Back to Page 1 button
        backButton = ttk.Button(self, text = "Back", command = lambda: controller.show_frame(HRCHCL_Page1))
        canvas.create_window(0, 45, window = backButton)

        # Back to Home button
        homeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        homeButton.pack(side = tk.BOTTOM, pady = 10)


class DBSCAN_Page1(tk.Frame):    # k-means page
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=400, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # Label: k-means algorithm title
        headingLabel = tk.Label(self, text = "DBSCAN algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = headingLabel)   

        # Entries
        def placeholder_text(e):
            chooseColumnsEntry3.delete(0, "end")
        def placeholder_text2(e):
            EPSEntry.delete(0, "end")
        def placeholder_text3(e):
            minSamplesEntry.delete(0, "end")

        global chooseColumnsEntry3, EPSEntry, minSamplesEntry
        chooseColumnsEntry3 = tk.Entry(self, width = 20)
        chooseColumnsEntry3.insert(0, "Choose columns to plot...")
        chooseColumnsEntry3.bind("<FocusIn>", placeholder_text)
        EPSEntry = tk.Entry(self, width = 20)
        EPSEntry.insert(0, "Enter EPS...")
        EPSEntry.bind("<FocusIn>", placeholder_text2)
        minSamplesEntry = tk.Entry(self, width = 20)
        minSamplesEntry.insert(0, "Enter # of min samples...")
        minSamplesEntry.bind("<FocusIn>", placeholder_text3)
        
        # Functions
        def getImportData():
            global df
            import_file_path = filedialog.askopenfilename()
            df = pd.read_csv(import_file_path)

            if(_DEBUG == True):
                print(df.head())
                print("Row Count: " + str(len(df)), end = "\n")
                print("Col Count: " + str(len(df.axes[1])), end = "\n")
            
            global colNamesString
            colNamesString = ""
            for i in range(len(df.axes[1])):
                colNamesString += str(df.columns[i]) if(i == len(df.axes[1]) - 1) else str(df.columns[i]) + ", "
            if(_DEBUG == True): print("colString: " + colNamesString + '\n')

            # Column Names Label
            colNamesLabel = tk.Label(self, text = "Columns: " + colNamesString, font = ("Times", 14))
            canvas.create_window(325, 150, window = colNamesLabel)

            # Number of Columns Label
            rowCountLabel = tk.Label(self, text = "Row Count: " + str(len(df.axes[0])), font = ("Times", 14))
            canvas.create_window(325, 180, window = rowCountLabel)

            # EPS & Min Samples Entry
            canvas.create_window(325, 220, window = EPSEntry)
            canvas.create_window(325, 250, window = minSamplesEntry)

            if(len(df.axes[1]) >= 3):
                canvas.create_window(325, 280, window = chooseColumnsEntry3)
                if(len(df.axes[1]) >= 4):
                    PCA_notif_label = tk.Label(self, text = "*PCA will be used for dimension reduction (4+ col)", font = ("Times", 10))
                    canvas.create_window(325, 340, window = PCA_notif_label)
  
            # Next Button
            GoToDBSCANP2Button = tk.Button(self, text = "Next", command=lambda: controller.show_frame(DBSCAN_Page2), bg = "blue", relief = RAISED)
            canvas.create_window(325, 375, window = GoToDBSCANP2Button)
                
        # Import Excel Data Button
        browseButtonExcel = tk.Button(self, text = "Import Data (.csv)", command = getImportData, bg = "blue", relief = RAISED)
        canvas.create_window(325, 80, window = browseButtonExcel)

        # Back to Home button
        backToHomeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        backToHomeButton.pack(side = tk.BOTTOM, pady = 10)


class DBSCAN_Page2(tk.Frame):    # k-means page
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create Canvas
        canvas = tk.Canvas(self, width=650, height=120, relief=RAISED)
        canvas.pack(padx = 10, pady = 10)

        # k-means algorithm title label
        label1 = tk.Label(self, text = "DBSCAN algorithm clustering.", font = ("Helvetica", 30))
        canvas.create_window(325, 40, window = label1)   
        
        def runDBSCAN():
            userChoiceArray = list(df.columns) if(len(df.axes[1]) == 2) else chooseColumnsEntry3.get().replace(" ", "").split(',')
            figure = plt.Figure(figsize = (4,3), dpi = 100)

            if(len(userChoiceArray) == 2 or len(userChoiceArray) == 3):
                dbDefault = DBSCAN(eps = float(EPSEntry.get()), min_samples = int(minSamplesEntry.get())).fit(df[userChoiceArray])

            if(len(userChoiceArray) == 2):
                ax = figure.add_subplot(111)
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], c = dbDefault.labels_, s = 80, alpha = 0.5)
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)

            elif(len(userChoiceArray) == 3):
                ax = figure.add_subplot(111, projection='3d')
                ax.scatter(df[userChoiceArray[0]], df[userChoiceArray[1]], df[userChoiceArray[2]], c = dbDefault.labels_, s = 80, alpha = 0.5)
                ax.set_xlabel(userChoiceArray[0], fontsize = 10)
                ax.set_ylabel(userChoiceArray[1], fontsize = 10)
                ax.set_zlabel(userChoiceArray[2], fontsize = 10)
            
            elif(len(userChoiceArray) == 4):
                x = StandardScaler().fit_transform(df[userChoiceArray])      
                pca = PCA(n_components = 2)                  
                principal_components = pca.fit_transform(x)  
                principal_df = pd.DataFrame(data = principal_components, columns = ['pca1', 'pca2'])
                
                dbDefault = DBSCAN(eps = int(EPSEntry.get()), min_samples = int(minSamplesEntry.get())).fit(principal_df)
                ax = figure.add_subplot(111)
                ax.scatter(principal_df["pca1"], principal_df["pca2"], c = dbDefault.labels_.astype(float), s = 80, alpha = 0.5)
                ax.set_xlabel("Principal Component 1", fontsize = 10)
                ax.set_ylabel("Principal Component 2", fontsize = 10)
           
            # Pack figure
            scatter1 = FigureCanvasTkAgg(figure, self)
            scatter1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            toolbar = NavigationToolbar2Tk(scatter1, self)
            toolbar.update()
            scatter1._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
           
        # Buttons
        # Process Button
        processButton = tk.Button(self, text = "Run DBSCAN ▶", command = runDBSCAN)
        canvas.create_window(325, 80, window = processButton)

        # Back to Page 1 button
        backButton = ttk.Button(self, text = "Back", command = lambda: controller.show_frame(DBSCAN_Page1))
        canvas.create_window(0, 45, window = backButton)

        # Back to Home button
        homeButton = ttk.Button(self, text = "Back To Home", command = lambda: controller.show_frame(StartPage))
        homeButton.pack(side = tk.BOTTOM, pady = 10)

app = ClusterAnalysisApp()
centerWindow(app)
app.mainloop()