from rest_framework import serializers
from .models import FinancialGoal, GoalMilestone, GoalContribution

class GoalMilestoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = GoalMilestone
        fields = [
            'id', 'goal', 'title', 'description', 'target_date',
            'target_amount', 'is_completed', 'completed_at'
        ]
        read_only_fields = ['is_completed', 'completed_at']

class GoalContributionSerializer(serializers.ModelSerializer):
    class Meta:
        model = GoalContribution
        fields = [
            'id', 'goal', 'amount', 'date', 'description',
            'transaction', 'created_at'
        ]
        read_only_fields = ['created_at']

class FinancialGoalSerializer(serializers.ModelSerializer):
    milestones = GoalMilestoneSerializer(many=True, read_only=True)
    contributions = GoalContributionSerializer(many=True, read_only=True)
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)

    class Meta:
        model = FinancialGoal
        fields = [
            'id', 'user', 'household', 'type', 'type_display',
            'name', 'description', 'target_amount', 'current_amount',
            'start_date', 'target_date', 'status', 'status_display',
            'priority', 'priority_display', 'progress_percentage',
            'milestones', 'contributions', 'tags', 'attachments',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['user', 'current_amount', 'progress_percentage', 'created_at', 'updated_at']

class GoalProgressUpdateSerializer(serializers.Serializer):
    current_amount = serializers.DecimalField(max_digits=15, decimal_places=2)
    status = serializers.ChoiceField(choices=FinancialGoal.STATUS_CHOICES, required=False)

class GoalContributionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = GoalContribution
        fields = ['goal', 'amount', 'date', 'description', 'transaction']

    def validate(self, data):
        goal = data['goal']
        amount = data['amount']
        
        # Verificar que la contribución no exceda el monto objetivo
        if goal.current_amount + amount > goal.target_amount:
            raise serializers.ValidationError(
                "La contribución excedería el monto objetivo del goal"
            )
        
        return data 