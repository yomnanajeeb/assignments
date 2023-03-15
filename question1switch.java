import java.util.Scanner;

public class question1switch {
    public static void main(String[] args){

        Scanner scanner =new Scanner(System.in);
        System.out.println("please enter number");
        int x = scanner.nextInt();
            switch (x % 2) {
                case 0:
                    System.out.println( " is an even number.");
                    break;
                case 1:
                    System.out.println( " is an odd number.");
                    break;
        }

    }
}