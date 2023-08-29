# Decision_tree
Implementación de una técnica de aprendizaje máquina sin el uso de un framework

Decision Tree Algorithm

* El inicio del código consiste en la dfinición de funciones.
* A partir de la línea 210, comienza la cargar de datos. En este caso uso una base de datos para clasificar en obesidad o no obesidad. Aquí se puede modificar la base de datos si se desea.
* Posteriormente se corre el modelo, es importante definir dentro de la función de train train_tree() que se haga drop al target.
* Se hacen 4 corridas con diferente semilla a la hora de hacer el split de datos. De esta manera, se puede observar los diferentes accuracys para cada corrida con selección random de elementos de los datasets.
* Para correr el modelo, solo se debe de presionar en run y se imprimirán los accuracys.
