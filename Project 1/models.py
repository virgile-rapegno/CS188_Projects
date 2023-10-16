import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        trained = False
        while not trained:
            hadMistake = False
            for x, true_y in dataset.iterate_once(1):
                evaluated_y = self.get_prediction(x)
                if evaluated_y != nn.as_scalar(true_y):
                    hadMistake = True
                    self.get_weights().update(x, nn.as_scalar(true_y))
            if not hadMistake:
                trained = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here

        # 1er étage
        self.w1 = nn.Parameter(1, 50)
        self.b1 = nn.Parameter(1, 50)
        # 2nd étage
        self.w2 = nn.Parameter(50, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        temp = nn.Linear(x, self.w1)
        temp = nn.AddBias(temp, self.b1)
        temp = nn.ReLU(temp)

        temp = nn.Linear(temp, self.w2)
        temp = nn.AddBias(temp, self.b2)

        return temp

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        LEARNING_RATE = -0.01

        for x, y in dataset.iterate_forever(100):
            loss = self.get_loss(x, y)
            # On utilise le même critère de l'autograder pour être sûr de valider
            # C'est à dire qu'on souhaite avoir une erreur de 2% sur l'ensemble
            if (
                nn.as_scalar(
                    self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
                )
                <= 0.02
            ):
                break

            grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(
                loss, [self.w1, self.w2, self.b1, self.b2]
            )
            self.w1.update(grad_wrt_w1, LEARNING_RATE)
            self.w2.update(grad_wrt_w2, LEARNING_RATE)
            self.b1.update(grad_wrt_b1, LEARNING_RATE)
            self.b2.update(grad_wrt_b2, LEARNING_RATE)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        # Nombre de nodes à chaque étage afin de faire varier les paramètres
        # et trouver une version qui fonctionne
        nbNode1 = 300
        nbNode2 = 100
        nbNode3 = 10

        # 1er étage
        self.w1 = nn.Parameter(784, nbNode1)
        self.b1 = nn.Parameter(1, nbNode1)
        # 2ème étage
        self.w2 = nn.Parameter(nbNode1, nbNode2)
        self.b2 = nn.Parameter(1, nbNode2)
        # 3ème étage
        self.w3 = nn.Parameter(nbNode2, nbNode3)
        self.b3 = nn.Parameter(1, nbNode3)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        temp = nn.Linear(x, self.w1)
        temp = nn.AddBias(temp, self.b1)
        temp = nn.ReLU(temp)

        temp = nn.Linear(temp, self.w2)
        temp = nn.AddBias(temp, self.b2)
        temp = nn.ReLU(temp)

        temp = nn.Linear(temp, self.w3)
        temp = nn.AddBias(temp, self.b3)

        return temp

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        LEARNING_RATE = -0.7

        for x, y in dataset.iterate_forever(30):
            loss = self.get_loss(x, y)
            validation = dataset.get_validation_accuracy()

            # On fait évoluer le paramètre d'apprentissage afin de
            # ne pas modifier trop fortement le modèle sur la fin
            if validation > 0.975:
                break
            elif validation > 0.97:
                LEARNING_RATE = -0.05
            elif validation > 0.96:
                LEARNING_RATE = -0.1
            elif validation > 0.95:
                LEARNING_RATE = -0.2
            elif validation > 0.9:
                LEARNING_RATE = -0.5

            (
                grad_wrt_w1,
                grad_wrt_w2,
                grad_wrt_w3,
                grad_wrt_b1,
                grad_wrt_b2,
                grad_wrt_b3,
            ) = nn.gradients(
                loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
            )
            self.w1.update(grad_wrt_w1, LEARNING_RATE)
            self.w2.update(grad_wrt_w2, LEARNING_RATE)
            self.w3.update(grad_wrt_w3, LEARNING_RATE)
            self.b1.update(grad_wrt_b1, LEARNING_RATE)
            self.b2.update(grad_wrt_b2, LEARNING_RATE)
            self.b3.update(grad_wrt_b3, LEARNING_RATE)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        # Profondeur de la couche de neurone
        d = 200

        self.w = nn.Parameter(self.num_chars, d)

        self.w1 = nn.Parameter(d, d)
        self.b1 = nn.Parameter(1, d)

        self.w_hidden = nn.Parameter(d, d)

        self.w_language = nn.Parameter(d, 5)
        self.b_language = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # F_init
        h = nn.Linear(xs[0], self.w)
        h = nn.ReLU(h)
        h = nn.Linear(h, self.w1)
        h = nn.AddBias(h, self.b1)

        # F
        for x in xs[1:]:
            h = nn.Add(nn.Linear(x, self.w), nn.Linear(h, self.w_hidden))
            h = nn.ReLU(h)
            h = nn.Linear(h, self.w1)
            h = nn.AddBias(h, self.b1)

        # Post-processing pour se ramener à 5 catégories (langues)
        h = nn.Linear(h, self.w_language)
        h = nn.AddBias(h, self.b_language)

        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        LEARNING_RATE = -0.05
        COUNT = 0

        for x, y in dataset.iterate_forever(200):
            loss = self.get_loss(x, y)
            validation = dataset.get_validation_accuracy()

            # On veut un résultat consistant après un certain nombre de mises
            # à jour avant de confirmer que l'apprentissage est fini
            if validation > 0.83:
                COUNT += 1
            else:
                COUNT = 0

            if COUNT > 30:
                break

            (
                grad_wrt_w,
                grad_wrt_w1,
                grad_wrt_b1,
                grad_wrt_w_hidden,
                grad_wrt_b_language,
                grad_wrt_w_language,
            ) = nn.gradients(
                loss,
                [
                    self.w,
                    self.w1,
                    self.b1,
                    self.w_hidden,
                    self.b_language,
                    self.w_language,
                ],
            )
            self.w.update(grad_wrt_w, LEARNING_RATE)
            self.w1.update(grad_wrt_w1, LEARNING_RATE)
            self.b1.update(grad_wrt_b1, LEARNING_RATE)
            self.w_hidden.update(grad_wrt_w_hidden, LEARNING_RATE)
            self.b_language.update(grad_wrt_b_language, LEARNING_RATE)
            self.w_language.update(grad_wrt_w_language, LEARNING_RATE)
