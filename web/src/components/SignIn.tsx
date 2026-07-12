import { GoogleLogin } from '@react-oauth/google';
import { loginWithGoogle, type AuthUser } from '../lib/authClient';

interface SignInProps {
  onSignedIn: (user: AuthUser) => void;
}

export function SignIn({ onSignedIn }: SignInProps) {
  return (
    <div className="sign-in">
      <p>Sign in to start chatting. Your conversations and memories are tied to your account.</p>
      <GoogleLogin
        onSuccess={async (credentialResponse) => {
          if (!credentialResponse.credential) return;
          const user = await loginWithGoogle(credentialResponse.credential);
          onSignedIn(user);
        }}
        onError={() => console.error('Google sign-in failed')}
      />
    </div>
  );
}
