using Modding;
using InControl;

namespace HKRL.Game
{
	   public class InputDeviceShim : InputDevice
    {
		// private bool[] Movement = new bool[4] {false, false, false, false};
		private bool KeyUp = false;
		private bool KeyDown = false;
		private bool KeyLeft = false;
		private bool KeyRight = false;
		private bool KeyJump = false;
		private bool KeyAttack = false;
		private bool KeyDash = false;
		private bool KeyCast = false;


		// public bool[] Actions = [false, false, false, false, false];

        public InputDeviceShim() :
            base("CustomInputShimDevice")
        {
			AddControl(InputControlType.DPadUp, "Up");
			AddControl(InputControlType.DPadDown, "Down");
			AddControl(InputControlType.DPadLeft, "Left");
			AddControl(InputControlType.DPadRight, "Right");
			AddControl(InputControlType.Action1, "Jump");
			AddControl(InputControlType.Action2, "Cast"); // Hold for heal
			AddControl(InputControlType.Action3, "Attack");
			AddControl(InputControlType.Action4, "DreamNail");
			AddControl(InputControlType.RightTrigger, "Dash");
			AddControl(InputControlType.LeftTrigger, "SuperDash");
			AddControl(InputControlType.RightBumper, "QuickCast");
        }
		public override void Update(ulong updateTick, float deltaTime)
		{
			// base.Update(updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadUp, KeyUp, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadDown, KeyDown, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadLeft, KeyLeft, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadRight, KeyRight, updateTick, deltaTime);
			UpdateWithState(InputControlType.Action1, KeyJump, updateTick, deltaTime);

			UpdateWithState(InputControlType.RightBumper, KeyCast, updateTick, deltaTime);
			UpdateWithState(InputControlType.Action3, KeyAttack, updateTick, deltaTime);
			UpdateWithValue(InputControlType.RightTrigger, KeyDash ? 1 : 0, updateTick, deltaTime);

			// ResetState();
		}

		private static bool CanDash() =>
            ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDash");

        private static bool CanAttack() =>
            ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanAttack");
			
		private static bool CanJump() =>
            ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanJump");

        private static bool CanDoubleJump() =>
            ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDoubleJump");

		private static bool CanCast() => 
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanCast");


		private static bool CanWallJump() =>
            ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanWallJump");
		public void Reset() {
			KeyUp = false;
			KeyDown = false;
			KeyLeft = false;
			KeyRight = false;
			KeyJump = false;
			KeyAttack = false;
			KeyDash = false;
			KeyCast = false;
		}

		public void Dash() {
			if(!CanDash()) return;
			if (KeyLeft) {
				HeroController.instance.FaceLeft();
			} else if (KeyRight) {
				HeroController.instance.FaceRight();
			}

			KeyDash = true;
			KeyJump = false;
		}
		
		public void Left() {
			KeyLeft = true;
			KeyRight = false;
		}

		public void Right() {
			KeyRight = true;
			KeyLeft = false;
		}

		public void Up() {
			KeyUp = true;
			KeyDown = false;
		}

		public void Down() {
			KeyDown = true;
			KeyUp = false;
		}
		public void Jump() {
			if (!CanJump() && !CanDoubleJump() && !CanWallJump()) return;
			KeyJump = true;
			KeyDash = false;
		}

		public void Attack() {
			if (!CanAttack()) return;
			if (KeyLeft) {
				HeroController.instance.FaceLeft();
			}
			if (KeyRight) {
				HeroController.instance.FaceRight();
			}

			KeyAttack = true;
			KeyCast = false;
		}

		public void Cast() {
			if (!CanCast()) return;
			
			if (KeyLeft) {
				HeroController.instance.FaceLeft();
			}
			if (KeyRight) {
				HeroController.instance.FaceRight();
			}

			KeyCast = true;
			KeyAttack = false;
		}

		public void StopLR() {
			KeyLeft = false;
			KeyRight = false;
		}

		public void StopUD() {
			KeyUp = false;
			KeyDown = false;
		}

		public void StopJD() {
			KeyJump = false;
			KeyDash = false;
		}

		public void StopAC() {
			KeyAttack = false;
			KeyCast = false;
		}
    }

}