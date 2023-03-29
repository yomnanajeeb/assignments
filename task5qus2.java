public class task5qus2 {
    public static void main(String[] args) {
        // Using a for loop
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print("*");
            }
            System.out.println();
        }
        // Using a while loop
        int i = 1;
        while (i <= 5) {
            int j = 1;
            while (j <= i) {
                System.out.print("*");
                j++;
            }
            System.out.println();
            i++;
        }
        // Using a do-while loop
        int x= 1;
        do{
            int j = 1;
            do{
                System.out.print("*");
                j++;
            }while(j <= x);
            System.out.println();
            x++;
        }while(x <= 5);
    }
}